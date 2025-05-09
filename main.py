from config import ROOT, MEGAD_NAME, DEVICE, THRESHOLD
from src.transforms import transform, transforms_aliked, transform_display
from src.utils import create_sample_submission
from src.dataset import load_datasets
from src.matcher import build_megadescriptor, build_aliked, build_clip
from src.fusion import build_wildfusion

import timm
import numpy as np


def main():
    # 1. Load the full dataset
    dataset, dataset_db, dataset_query, dataset_calib = load_datasets(ROOT)

    # 2. Load MegaDescriptor model (global descriptor backbone)
    model = timm.create_model(MEGAD_NAME, num_classes=0, pretrained=True).to(DEVICE)

    # 3. Build matchers
    matcher_mega = build_megadescriptor(model=model, transform=transform, device=DEVICE)
    matcher_aliked = build_aliked(transform=transforms_aliked, device=DEVICE)
    matcher_clip = build_clip(device=DEVICE)

    # 4. Build fusion model and apply calibration
    fusion = build_wildfusion(
        dataset_calib, dataset_calib,
        matcher_aliked, matcher_mega, matcher_clip,
        priority_pipeline=matcher_mega
    )

    # 5. Compute predictions per query group (by dataset) but compare against full DB
    predictions_all = []
    image_ids_all = []

    # 6. Queary의 종별 전략을 다르게 적용해 비교
    for dataset_name in dataset_query.metadata["dataset"].unique():
        query_subset = dataset_query.get_subset(dataset_query.metadata["dataset"] == dataset_name)

        similarity = fusion(query_subset, dataset_db, B=25)

        # top‑1 / top‑2 probabilities (already calibrated)
        idx_sorted = similarity.argsort(axis=1)
        top_idx     = idx_sorted[:, -1]
        second_idx  = idx_sorted[:, -2]
        p_top1      = similarity[np.arange(len(query_subset)), top_idx]
        p_top2      = similarity[np.arange(len(query_subset)), second_idx]
        gap         = p_top1 - p_top2

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
