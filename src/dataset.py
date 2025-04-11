from torch.utils.data import Dataset
from PIL import Image
import os

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