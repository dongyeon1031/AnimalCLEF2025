# ====================
# 1. Imports & Utils
# ====================
import os
import numpy as np
import pandas as pd
import timm
import torchvision.transforms as T

from wildlife_datasets.datasets import AnimalCLEF2025
from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.similarity.wildfusion import SimilarityPipeline, WildFusion
from wildlife_tools.similarity.pairwise.lightglue import MatchLightGlue
from wildlife_tools.features.local import AlikedExtractor
from wildlife_tools.similarity.calibration import IsotonicCalibration

def create_sample_submission(dataset_query, predictions, file_name='sample_submission.csv'):
    df = pd.DataFrame({
        'image_id': dataset_query.metadata['image_id'],
        'identity': predictions
    })
    df.to_csv(file_name, index=False)

# ====================
# 2. Transforms
# ====================
root = r"C:\Users\user\Desktop\kimdongyeon\CV_proj\animal-clef-2025"

transform_display = T.Compose([
    T.Resize([384, 384]),
])
transform = T.Compose([
    *transform_display.transforms,
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225))
])
transforms_aliked = T.Compose([
    T.Resize([512, 512]),
    T.ToTensor()
])

# ====================
# 3. Load dataset
# ====================
dataset = AnimalCLEF2025(root, load_label=True)
dataset_database = dataset.get_subset(dataset.metadata['split'] == 'database')
dataset_query = dataset.get_subset(dataset.metadata['split'] == 'query')
dataset_calibration = AnimalCLEF2025(root, df=dataset_database.metadata[:100], load_label=True)
n_query = len(dataset_query)
print(f"Loaded {len(dataset_database)} database images and {n_query} query images.")

# ====================
# 4. Load models & matchers
# ====================
name = 'hf-hub:BVRA/MegaDescriptor-L-384'
model = timm.create_model(name, num_classes=0, pretrained=True)
device = 'cuda'

matcher_aliked = SimilarityPipeline(
    matcher = MatchLightGlue(features='aliked', device=device, batch_size=16),
    extractor = AlikedExtractor(),
    transform = transforms_aliked,
    calibration = IsotonicCalibration()
)

matcher_mega = SimilarityPipeline(
    matcher = CosineSimilarity(),
    extractor = DeepFeatures(model=model, device=device, batch_size=16),
    transform = transform,
    calibration = IsotonicCalibration()
)

# ====================
# 5. WildFusion calibration
# ====================
wildfusion = WildFusion(
    calibrated_pipelines=[matcher_aliked, matcher_mega],
    priority_pipeline=matcher_mega
)
wildfusion.fit_calibration(dataset_calibration, dataset_calibration)

# ====================
# 6. Compute similarity
# ====================
similarity = wildfusion(dataset_query, dataset_database, B=25)

# ====================
# 7. Top-1 prediction + threshold
# ====================
pred_idx = similarity.argsort(axis=1)[:, -1]
pred_scores = similarity[np.arange(n_query), pred_idx]

new_individual = 'new_individual'
threshold = 0.6
labels = dataset_database.labels_string
predictions = labels[pred_idx].copy()
predictions[pred_scores < threshold] = new_individual

# ====================
# 8. Save submission
# ====================
create_sample_submission(dataset_query, predictions, file_name='sample_submission.csv')
print("sample_submission.csv saved!")