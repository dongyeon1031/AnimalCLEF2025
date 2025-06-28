from config import ROOT, MEGAD_NAME, DEVICE, THRESHOLD
from src.transforms import transforms_aliked, transform_tta_mega
from src.utils import create_sample_submission
from src.dataset import load_datasets
from src.fusion import build_wildfusion
from src.matcher import build_megadescriptor, build_aliked, build_eva02
from src.fusion_head import FusionMLP
from src.utils import set_seed

import timm
import numpy as np


def main():
    set_seed(42)
    # 1. Load the full dataset
    dataset, dataset_db, dataset_query, dataset_calib = load_datasets(ROOT, calibration_size=1000)

    # 2. Load MegaDescriptor model (global descriptor backbone)
    model = timm.create_model(MEGAD_NAME, num_classes=0, pretrained=True).to(DEVICE)

    # 3. Build matchers
    matcher_mega = build_megadescriptor(model=model, transform=transform_tta_mega, device=DEVICE)
    matcher_aliked = build_aliked(transform=transforms_aliked, device=DEVICE)


    # 4. Build fusion model and apply calibration
    fusion = build_wildfusion(
        dataset_calib, dataset_calib,
        matcher_aliked, matcher_mega,
        priority_pipeline=matcher_mega
    )

    matcher_eva = build_eva02(device=DEVICE)
    fusion_head = FusionMLP().to(DEVICE)   # 랜덤 초기화 (미학습)

    dataset_db_eva = dataset_db.get_subset(slice(None))
    dataset_db_eva.transform = matcher_eva.transform   # 1-arg transform
    emb_db_eva = matcher_eva.extractor(dataset_db_eva)
    emb_db_eva = emb_db_eva / np.linalg.norm(emb_db_eva, axis=1, keepdims=True)

    # 5. Compute predictions per query group (by dataset) but compare against full DB
    predictions_all = []
    image_ids_all = []

    # 6. Queary의 종별 전략을 다르게 적용해 비교
    for dataset_name in dataset_query.metadata["dataset"].unique():
        query_subset = dataset_query.get_subset(  # ★ 먼저 정의
            dataset_query.metadata["dataset"] == dataset_name)

        # 1) WildFusion similarity (Mega+ALIKED+LoFTR)
        sim_fusion = fusion(query_subset, dataset_db, B=25)

        # 2) EVA02 cosine similarity
        emb_q_eva = matcher_eva.extractor(query_subset)
        emb_q_eva = emb_q_eva / np.linalg.norm(emb_q_eva, axis=1, keepdims=True)
        sim_eva = emb_q_eva @ emb_db_eva.T

        # 3) 가중 평균 (또는 향후 MLP)
        combined_sim = 0.5 * sim_fusion + 0.5 * sim_eva   # α=0.5 임시

        # --- 이후 모든 idx / score 계산을 combined_sim 기준으로 ---
        idx_sorted = combined_sim.argsort(axis=1)
        top_idx    = idx_sorted[:, -1]
        p_top1     = combined_sim[np.arange(len(query_subset)), top_idx]

        # second_idx  = idx_sorted[:, -2]
        # p_top2      = similarity[np.arange(len(query_subset)), second_idx]
        # gap         = p_top1 - p_top2

        # adaptive threshold: global + small offset
        base_th = THRESHOLD
        offsets = {"SeaTurtleID2022": -0.02, "SalamanderID2025": +0.02}
        thr = base_th + offsets.get(dataset_name, 0.0)

        labels = dataset_db.labels_string
        predictions = labels[top_idx].copy()
        predictions[(p_top1 < thr)] = "new_individual"

        predictions_all.extend(predictions)
        image_ids_all.extend(query_subset.metadata["image_id"])

    # 7. Save to CSV
    import pandas as pd
    df = pd.DataFrame({"image_id": image_ids_all, "identity": predictions_all})
    df.to_csv("sample_submission.csv", index=False)
    print("✅ sample_submission.csv saved!")


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
