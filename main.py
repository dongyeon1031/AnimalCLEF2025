# main.py

from config import ROOT, MEGAD_NAME, DEVICE, THRESHOLD
from src.transforms import transform, transforms_aliked
from src.utils import create_sample_submission
from src.dataset import load_datasets
from src.matcher import build_megadescriptor, build_aliked
from src.fusion import build_wildfusion

import timm
import numpy as np


def main():
    # 1. Load datasets
    dataset, dataset_db, dataset_query, dataset_calib = load_datasets(ROOT)
    n_query = len(dataset_query)

    # 2. Load backbone model
    model = timm.create_model(MEGAD_NAME, num_classes=0, pretrained=True).to(DEVICE)

    # 3. Build matchers
    matcher_mega = build_megadescriptor(model=model, transform=transform, device=DEVICE)
    matcher_aliked = build_aliked(transform=transforms_aliked, device=DEVICE)

    # 4. Build WildFusion & calibrate
    fusion = build_wildfusion(matcher_aliked, matcher_mega, dataset_calib, dataset_calib)

    # 5. Compute similarity
    similarity = fusion(dataset_query, dataset_db, B=25)

    # 6. Predict
    pred_idx = similarity.argsort(axis=1)[:, -1]
    pred_scores = similarity[np.arange(n_query), pred_idx]

    labels = dataset_db.labels_string
    predictions = labels[pred_idx].copy()
    predictions[pred_scores < THRESHOLD] = 'new_individual'

    # 7. Save
    create_sample_submission(dataset_query, predictions)
    print("✅ sample_submission.csv saved!")


if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
