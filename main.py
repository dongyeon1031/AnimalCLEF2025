import numpy as np
import pandas as pd
import torch
import timm

from config import ROOT, MEGAD_NAME, DEVICE, THRESHOLD
from src.dataset import load_datasets
from src.utils import set_seed
from src.transforms import transform_tta_mega, transforms_aliked

from src.factory import build_megadescriptor, build_aliked, build_eva02
from src.calibrators import XGBoostCalibrator
from src.base import UniversalPipeline

def main():
    set_seed(42)
    
    # ---------------------------------------------------------
    # 1. 데이터셋 로드
    # ---------------------------------------------------------
    print("[Main] Loading datasets...")
    dataset, dataset_db, dataset_query, dataset_calib = load_datasets(ROOT, calibration_size=1000)
    
    # [수정 1] 정수 인덱스 대신 '불리언 마스크' 사용 (전체 True)
    # Pandas KeyError 방지용
    db_mask = np.ones(len(dataset_db), dtype=bool)
    calib_mask = np.ones(len(dataset_calib), dtype=bool)

    # ---------------------------------------------------------
    # 2. 파이프라인 구축 (UniversalPipeline)
    # ---------------------------------------------------------
    print("[Main] Building pipelines...")

    # (1) MegaDescriptor Pipeline
    print(f" - MegaDescriptor ({MEGAD_NAME})")
    model_mega = timm.create_model(MEGAD_NAME, num_classes=0, pretrained=True).to(DEVICE)
    pipeline_mega = build_megadescriptor(model=model_mega, transform=transform_tta_mega, device=DEVICE)

    # Mega 전용 XGBoost 생성
    calibrator_mega = XGBoostCalibrator() 
    pipeline_mega = UniversalPipeline(scorer=base_mega.scorer, calibrator=calibrator_mega)

    # (2) ALIKED Pipeline
    print(f" - ALIKED (Local Features)")
    pipeline_aliked = build_aliked(transform=transforms_aliked, device=DEVICE)

    # ALIKED 전용 XGBoost 생성
    calibrator_aliked = XGBoostCalibrator()
    pipeline_aliked = UniversalPipeline(scorer=base_aliked.scorer, calibrator=calibrator_aliked)

    # (3) EVA02 Pipeline
    print(f" - EVA02 (Global Features)")
    pipeline_eva = build_eva02(device=DEVICE)

    # EVA02 전용 XGBoost 생성
    calibrator_eva = XGBoostCalibrator()
    pipeline_eva = UniversalPipeline(scorer=base_eva.scorer, calibrator=calibrator_eva)


    # ---------------------------------------------------------
    # 3. 캘리브레이션 (Calibration)
    # ---------------------------------------------------------
    print("[Main] Starting calibration...")

    # (1) Mega Calibration
    print(" -> Training Mega-XGBoost...")
    calib_mega = dataset_calib.get_subset(calib_mask)
    calib_mega.transform = transform_tta_mega
    pipeline_mega.fit_calibration(calib_mega, calib_mega)

    # (2) ALIKED Calibration
    print(" -> Training ALIKED-XGBoost...")
    calib_aliked = dataset_calib.get_subset(calib_mask)
    calib_aliked.transform = transforms_aliked
    pipeline_aliked.fit_calibration(calib_aliked, calib_aliked)

    # (3) EVA02 Calibration
    print(" -> Training EVA-XGBoost...")
    calib_eva = dataset_calib.get_subset(calib_mask)
    calib_eva.transform = pipeline_eva.transform
    pipeline_eva.fit_calibration(calib_eva, calib_eva)


    # ---------------------------------------------------------
    # 4. 추론 (Inference) & 앙상블
    # ---------------------------------------------------------
    predictions_all = []
    image_ids_all = []
    
    # DB 데이터셋 준비 (각 모델별 Transform 적용)
    db_mega = dataset_db.get_subset(db_mask)
    db_mega.transform = transform_tta_mega
    
    db_aliked = dataset_db.get_subset(db_mask)
    db_aliked.transform = transforms_aliked
    
    db_eva = dataset_db.get_subset(db_mask)
    db_eva.transform = pipeline_eva.transform

    print("[Main] Processing queries by dataset...")
    for dataset_name in dataset_query.metadata["dataset"].unique():
        query_mask = dataset_query.metadata["dataset"] == dataset_name
        query_subset = dataset_query.get_subset(query_mask)
        
        print(f" -> Processing {dataset_name} ({len(query_subset)} images)...")
        
        # 쿼리 부분집합용 전체 마스크 (subset 내부에서 전체 선택)
        subset_full_mask = np.ones(len(query_subset), dtype=bool)

        # -----------------------------------------------------
        # Step A: MegaDescriptor (Global Search)
        # -----------------------------------------------------
        # 쿼리에 Mega Transform 적용
        query_mega = query_subset.get_subset(subset_full_mask)
        query_mega.transform = transform_tta_mega
        
        # 점수 계산
        probs_mega = pipeline_mega(query_mega, db_mega)


        # -----------------------------------------------------
        # Step B: ALIKED Reranking (Local Refinement)
        # -----------------------------------------------------
        # Mega 점수 기준 Top-K 선정
        B = 25
        topk_vals, topk_indices = torch.topk(torch.from_numpy(probs_mega), k=min(B, probs_mega.shape[1]), dim=1)
        
        pairs = []
        rows = np.arange(len(query_subset))
        for r, cols in zip(rows, topk_indices.numpy()):
            for c in cols:
                pairs.append((r, c))
        
        query_aliked = query_subset.get_subset(subset_full_mask)
        query_aliked.transform = transforms_aliked
        
        # ALIKED도 이제 '확률'을 반환합니다.
        probs_aliked_sparse = pipeline_aliked(query_aliked, db_aliked, pairs=pairs)
        
        # 전체 행렬 만들기 (기본값은 0, 즉 확률 0%로 설정)
        probs_aliked_full = np.zeros_like(probs_mega)
        
        if probs_aliked_sparse.ndim == 1:
            q_idxs = [p[0] for p in pairs]
            db_idxs = [p[1] for p in pairs]
            probs_aliked_full[q_idxs, db_idxs] = probs_aliked_sparse
        else:
            probs_aliked_full = probs_aliked_sparse

        # Fusion: Mega + ALIKED
        probs_fusion = probs_mega.copy()
        valid_mask = (probs_aliked_full > 0)
        probs_fusion[valid_mask] = (probs_mega[valid_mask] + probs_aliked_full[valid_mask]) / 2


        # -----------------------------------------------------
        # Step C: EVA02 (Global Ensemble)
        # -----------------------------------------------------
        query_eva = query_subset.get_subset(subset_full_mask)
        query_eva.transform = pipeline_eva.transform
        
        probs_eva = pipeline_eva(query_eva, db_eva)


        # -----------------------------------------------------
        # Step D: Final Ensemble
        # -----------------------------------------------------
        final_probs = 0.5 * probs_fusion + 0.5 * probs_eva
        
        
        # -----------------------------------------------------
        # Step E: Prediction & Thresholding
        # -----------------------------------------------------
        idx_sorted = final_probs.argsort(axis=1)
        top_idx = idx_sorted[:, -1]
        p_top1 = final_probs[np.arange(len(query_subset)), top_idx]

        thr = THRESHOLD
        
        pred_labels = dataset_db.labels_string[top_idx].copy()
        pred_labels[p_top1 < thr] = "new_individual"
        
        predictions_all.extend(pred_labels)
        image_ids_all.extend(query_subset.metadata["image_id"])

    # 7. Save to CSV
    import pandas as pd
    df = pd.DataFrame({"image_id": image_ids_all, "identity": predictions_all})
    df.to_csv("submission_all_xgboost.csv", index=False)
    print("✅ All models + XGBoost experiment finished! Saved to submission_all_xgboost.csv")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
