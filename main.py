import numpy as np
import pandas as pd
import torch
import timm

from config import ROOT, MEGAD_NAME, DEVICE, THRESHOLD
from src.dataset import load_datasets
from src.utils import set_seed
from src.transforms import transform_tta_mega, transforms_aliked

from src.factory import build_megadescriptor, build_aliked, build_eva02

def main():
    set_seed(42)
    
    # ---------------------------------------------------------
    # 1. 데이터셋 로드
    # ---------------------------------------------------------
    print("[Main] Loading datasets...")
    dataset, dataset_db, dataset_query, dataset_calib = load_datasets(ROOT, calibration_size=1000)
    
    # DB 전체 인덱스 (Transform 적용 시 사본 생성을 위해 사용)
    db_indices = np.arange(len(dataset_db))
    calib_indices = np.arange(len(dataset_calib))

    # ---------------------------------------------------------
    # 2. 파이프라인 구축 (UniversalPipeline)
    # ---------------------------------------------------------
    print("[Main] Building pipelines...")

    # (1) MegaDescriptor Pipeline
    print(f" - MegaDescriptor ({MEGAD_NAME})")
    model_mega = timm.create_model(MEGAD_NAME, num_classes=0, pretrained=True).to(DEVICE)
    pipeline_mega = build_megadescriptor(model=model_mega, transform=transform_tta_mega, device=DEVICE)

    # (2) ALIKED Pipeline
    print(f" - ALIKED (Local Features)")
    pipeline_aliked = build_aliked(transform=transforms_aliked, device=DEVICE)

    # (3) EVA02 Pipeline
    print(f" - EVA02 (Global Features)")
    pipeline_eva = build_eva02(device=DEVICE)


    # ---------------------------------------------------------
    # 3. 캘리브레이션 (Calibration)
    #    각 모델에 맞는 Transform을 적용한 데이터셋으로 학습
    # ---------------------------------------------------------
    print("[Main] Starting calibration...")

    # (1) Mega Calibration
    calib_mega = dataset_calib.get_subset(calib_indices)
    calib_mega.transform = transform_tta_mega  # Mega 전용 Transform
    pipeline_mega.fit_calibration(calib_mega, calib_mega)

    # (2) ALIKED Calibration
    calib_aliked = dataset_calib.get_subset(calib_indices)
    calib_aliked.transform = transforms_aliked  # ALIKED 전용 Transform
    pipeline_aliked.fit_calibration(calib_aliked, calib_aliked)

    # (3) EVA02 Calibration
    calib_eva = dataset_calib.get_subset(calib_indices)
    calib_eva.transform = pipeline_eva.transform # EVA 전용 Transform (Factory에서 설정됨)
    pipeline_eva.fit_calibration(calib_eva, calib_eva)


    # ---------------------------------------------------------
    # 4. 추론 (Inference) & 앙상블
    # ---------------------------------------------------------
    predictions_all = []
    image_ids_all = []
    
    # DB 데이터셋 준비 (각 모델별 Transform 적용)
    db_mega = dataset_db.get_subset(db_indices)
    db_mega.transform = transform_tta_mega
    
    db_aliked = dataset_db.get_subset(db_indices)
    db_aliked.transform = transforms_aliked
    
    db_eva = dataset_db.get_subset(db_indices)
    db_eva.transform = pipeline_eva.transform

    print("[Main] Processing queries by dataset...")
    for dataset_name in dataset_query.metadata["dataset"].unique():
        # 해당 데이터셋의 쿼리만 추출
        query_indices = np.where(dataset_query.metadata["dataset"] == dataset_name)[0]
        query_subset = dataset_query.get_subset(query_indices)
        
        print(f" -> Processing {dataset_name} ({len(query_subset)} images)...")

        # -----------------------------------------------------
        # Step A: MegaDescriptor (Global Search)
        # -----------------------------------------------------
        # 쿼리에 Mega Transform 적용
        query_mega = query_subset.get_subset(np.arange(len(query_subset)))
        query_mega.transform = transform_tta_mega
        
        # 점수 계산 (보정된 확률값 반환)
        scores_mega = pipeline_mega(query_mega, db_mega)


        # -----------------------------------------------------
        # Step B: ALIKED Reranking (Local Refinement)
        # -----------------------------------------------------
        # Mega 점수 기준 Top-K 선정 (Shortlisting)
        B = 25  # Rerank batch size
        topk_vals, topk_indices = torch.topk(torch.from_numpy(scores_mega), k=min(B, scores_mega.shape[1]), dim=1)
        
        # ALIKED 계산을 위한 Pair 리스트 생성
        pairs = []
        rows = np.arange(len(query_subset))
        for r, cols in zip(rows, topk_indices.numpy()):
            for c in cols:
                pairs.append((r, c))
        
        # 쿼리에 ALIKED Transform 적용
        query_aliked = query_subset.get_subset(np.arange(len(query_subset)))
        query_aliked.transform = transforms_aliked
        
        # 선택된 Pair에 대해서만 ALIKED 점수 계산
        scores_aliked_sparse = pipeline_aliked(query_aliked, db_aliked, pairs=pairs)
        
        # Sparse 점수를 전체 매트릭스로 변환 (-inf로 초기화)
        scores_aliked_full = np.full_like(scores_mega, -np.inf)
        
        # 계산된 위치에 값 채워넣기
        # scores_aliked_sparse가 1D array라고 가정 (BaseScorer 구현에 따름)
        if scores_aliked_sparse.ndim == 1:
            q_idxs = [p[0] for p in pairs]
            db_idxs = [p[1] for p in pairs]
            scores_aliked_full[q_idxs, db_idxs] = scores_aliked_sparse
        else:
            # Scorer가 전체 매트릭스를 리턴했다면 그대로 사용 (비효율적이지만 안전)
            scores_aliked_full = scores_aliked_sparse

        # Fusion: Mega + ALIKED
        # ALIKED 점수가 없는 곳(-inf)은 Mega 점수도 무시되거나 낮아지게 됨
        # 확률 평균 (Mega는 전체, ALIKED는 Top-K만 유효)
        scores_fusion = scores_mega.copy()
        valid_mask = (scores_aliked_full > -999) # -inf 체크
        scores_fusion[valid_mask] = (scores_mega[valid_mask] + scores_aliked_full[valid_mask]) / 2


        # -----------------------------------------------------
        # Step C: EVA02 (Global Ensemble)
        # -----------------------------------------------------
        # 쿼리에 EVA Transform 적용
        query_eva = query_subset.get_subset(np.arange(len(query_subset)))
        query_eva.transform = pipeline_eva.transform
        
        # 점수 계산 (보정된 확률값)
        scores_eva = pipeline_eva(query_eva, db_eva)


        # -----------------------------------------------------
        # Step D: Final Ensemble
        # -----------------------------------------------------
        # Fusion(Mega+Aliked) 점수와 EVA 점수 결합
        final_scores = 0.5 * scores_fusion + 0.5 * scores_eva
        
        
        # -----------------------------------------------------
        # Step E: Prediction & Thresholding
        # -----------------------------------------------------
        # Top-1 찾기
        idx_sorted = final_scores.argsort(axis=1)
        top_idx = idx_sorted[:, -1]
        p_top1 = final_scores[np.arange(len(query_subset)), top_idx]

        # Adaptive Threshold 적용
        thr = THRESHOLD # 필요 시 dataset_name에 따라 조정 가능
        
        pred_labels = dataset_db.labels_string[top_idx].copy()
        pred_labels[p_top1 < thr] = "new_individual"
        
        predictions_all.extend(pred_labels)
        image_ids_all.extend(query_subset.metadata["image_id"])

    # ---------------------------------------------------------
    # 5. 결과 저장
    # ---------------------------------------------------------
    df = pd.DataFrame({"image_id": image_ids_all, "identity": predictions_all})
    df.to_csv("sample_submission.csv", index=False)
    print("✅ sample_submission.csv saved!")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()