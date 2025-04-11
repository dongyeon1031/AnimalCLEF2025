import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import timm

# ---------------------- 설정 ----------------------
ROOT_DIR = 'C:/your/path/to/animal-clef-2025'
IMAGE_DIR = os.path.join(ROOT_DIR, 'images')
METADATA_PATH = os.path.join(ROOT_DIR, 'metadata.csv')

transform_display = T.Compose([
    T.Resize([384, 384]),
])
transform = T.Compose([
    *transform_display.transforms,
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225))
])

# ---------------------- Dataset 클래스 ----------------------
class AnimalClefDataset(Dataset):
    def __init__(self, metadata_df, root_dir, transform=None):
        self.metadata = metadata_df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        img_path = os.path.join(self.root_dir, row['path'])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, row['identity'], row['image_id']

# ---------------------- Feature 추출 클래스 ----------------------
class DeepFeatures:
    def __init__(self, model, device='cuda', batch_size=32, num_workers=0):
        self.model = model.to(device).eval()
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __call__(self, dataset):
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        all_features = []
        image_ids = []

        with torch.no_grad():
            for images, _, image_id_batch in loader:
                images = images.to(self.device)
                feats = self.model(images)
                feats = F.normalize(feats, dim=-1)
                all_features.append(feats.cpu())
                image_ids.extend(image_id_batch)

        all_features = torch.cat(all_features, dim=0)
        return {'features': all_features, 'image_ids': image_ids}

# wildlife-tools 원본 사용 시 대체 가능
# from wildlife_tools.features import DeepFeatures as WildlifeDeepFeatures
# extractor = WildlifeDeepFeatures(model, device=device)

# ---------------------- Cosine Similarity ----------------------
class CosineSimilarity:
    def __call__(self, features_query, features_database):
        q = features_query['features']
        d = features_database['features']
        similarity_matrix = q @ d.T
        return similarity_matrix

# ---------------------- Sample Submission ----------------------
def create_sample_submission(dataset_query, predictions, file_name='sample_submission.csv'):
    df = pd.DataFrame({
        'image_id': dataset_query.metadata['image_id'],
        'identity': predictions
    })
    df.to_csv(file_name, index=False)

# ---------------------- 실행부 ----------------------
if __name__ == '__main__':
    metadata = pd.read_csv(METADATA_PATH)
    metadata['filepath'] = metadata['path']  # 편의상 복사

    database_df = metadata[metadata['split'] == 'database'].copy()
    query_df = metadata[metadata['split'] == 'query'].copy()

    database_dataset = AnimalClefDataset(database_df, root_dir=ROOT_DIR, transform=transform)
    query_dataset = AnimalClefDataset(query_df, root_dir=ROOT_DIR, transform=transform)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = 'hf-hub:BVRA/MegaDescriptor-L-384'
    model = timm.create_model(model_name, num_classes=0, pretrained=True)

    extractor = DeepFeatures(model, device=device, batch_size=32, num_workers=0)
    features_database = extractor(database_dataset)
    features_query = extractor(query_dataset)

    similarity = CosineSimilarity()(features_query, features_database)

    n_query = similarity.shape[0]
    pred_idx = similarity.argsort(dim=1)[:, -1]
    pred_scores = similarity[torch.arange(n_query), pred_idx]

    labels = database_dataset.metadata['identity'].tolist()
    predictions = [labels[i] for i in pred_idx.tolist()]

    threshold = 0.6
    new_individual = 'new_individual'
    predictions = [
        pred if score >= threshold else new_individual
        for pred, score in zip(predictions, pred_scores)
    ]

    create_sample_submission(query_dataset, predictions, file_name='sample_submission.csv')