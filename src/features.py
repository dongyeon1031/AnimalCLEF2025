import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

class DeepFeatures:
    def __init__(self, model, device='cuda', batch_size=32, num_workers=0):
        self.model = model.to(device).eval()
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers

    def __call__(self, dataset):
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        all_features, image_ids = [], []
        with torch.no_grad():
            for images, _, image_id_batch in loader:
                images = images.to(self.device)
                feats = self.model(images)
                feats = F.normalize(feats, dim=-1)
                all_features.append(feats.cpu())
                image_ids.extend(image_id_batch)
        return {'features': torch.cat(all_features, dim=0), 'image_ids': image_ids}

class CosineSimilarity:
    def __call__(self, features_query, features_database):
        return features_query['features'] @ features_database['features'].T