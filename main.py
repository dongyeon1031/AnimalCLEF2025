from config import ROOT, MEGAD_NAME, DEVICE, THRESHOLD
from src.transforms import transform, transforms_aliked
from src.utils import create_sample_submission
from src.dataset import load_datasets
from src.matcher import build_megadescriptor, build_aliked
from src.fusion import build_wildfusion

import timm
import numpy as np

import os
import pandas as pd
import numpy as np
import cv2
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import kornia
from PIL import Image

# 📁 경로 설정 (ROOT는 config.py에서 import됨)
PROCESSED_DIR = os.path.join(ROOT, "processed")
METADATA_PATH = os.path.join(ROOT, "metadata.csv")

# ✨ CLAHE 적용 함수
def apply_clahe(img):
    img_np = np.array(img)
    if len(img_np.shape) == 2:  # Grayscale
        img_np = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img_np)
    else:  # RGB
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        img_np = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return Image.fromarray(img_np)

# ✨ Gamma correction
def apply_gamma(img, gamma=1.0):
    return TF.adjust_gamma(img, gamma)

# ✨ Gaussian smoothing
def apply_smoothing(img, sigma=1.0):
    img_tensor = T.ToTensor()(img).unsqueeze(0)
    smoothed = kornia.filters.gaussian_blur2d(img_tensor, (5, 5), (sigma, sigma))
    return T.ToPILImage()(smoothed.squeeze(0))

# ✨ 해상도 normalize (긴 변 384)
def resize_longest_side(img, target_size=384):
    w, h = img.size
    if w >= h:
        new_w = target_size
        new_h = int(target_size * h / w)
    else:
        new_h = target_size
        new_w = int(target_size * w / h)
    return img.resize((new_w, new_h), Image.BILINEAR)

# ✨ 종별 전처리 함수
def preprocess_image(image, species_name):
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image.")

    image = resize_longest_side(image, target_size=384)

    if species_name == "LynxID2025":
        image = apply_gamma(image, gamma=0.8)
        image = apply_clahe(image)
    elif species_name == "SalamanderID2025":
        image = apply_smoothing(image, sigma=1.0)
    elif species_name == "SeaTurtleID2022":
        image = apply_clahe(image)

    return image

# ✨ 전처리 실행 함수
def run_preprocessing():
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    metadata = pd.read_csv(METADATA_PATH)
    for idx, row in metadata.iterrows():
        img_path = os.path.join(ROOT, row["path"])
        species_name = row["dataset"]
        image_id = row["image_id"]

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"❌ Failed to load {img_path}: {e}")
            continue

        processed_img = preprocess_image(img, species_name)
        save_path = os.path.join(PROCESSED_DIR, f"{image_id}.png")
        processed_img.save(save_path)

        if idx % 500 == 0:
            print(f"✅ Processed {idx}/{len(metadata)} images")


def main():
    # 1. Load the full dataset
    if not os.path.exists(PROCESSED_DIR):
        run_preprocessing()
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
