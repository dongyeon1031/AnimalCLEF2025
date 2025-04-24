from config import ROOT, MEGAD_NAME, DEVICE, THRESHOLD
from src.transforms import transform, transforms_aliked
from src.utils import create_sample_submission
from src.dataset import load_datasets
from src.matcher import build_megadescriptor, build_aliked
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

    # 4. Build fusion model and apply calibration
    fusion = build_wildfusion(matcher_aliked, matcher_mega, dataset_calib, dataset_calib)

    # 5. Compute predictions per query group (by dataset) but compare against full DB
    predictions_all = []
    image_ids_all = []

    # 6. Queary의 종별 전략을 다르게 적용해 비교
    for dataset_name in dataset_query.metadata["dataset"].unique():
        query_subset = dataset_query.get_subset(dataset_query.metadata["dataset"] == dataset_name)

        # Apply per-dataset strategy: adjust threshold for LynxID2025
        threshold = THRESHOLD
        if dataset_name == "LynxID2025":
            # 시라소니 전략
            threshold = 0.35
        elif dataset_name == "SeaTurtleID2022":
            # 바다거북 전략
            threshold = 0.35
        elif dataset_name == "SalamanderID2025":
            # 도롱뇽 전략
            threshold = 0.35

        similarity = fusion(query_subset, dataset_db, B=25)
        pred_idx = similarity.argsort(axis=1)[:, -1]
        pred_scores = similarity[np.arange(len(query_subset)), pred_idx]

        labels = dataset_db.labels_string
        predictions = labels[pred_idx].copy()
        predictions[pred_scores < threshold] = 'new_individual'

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
