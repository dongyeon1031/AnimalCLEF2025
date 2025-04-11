import pandas as pd
from torchvision import transforms
from src.dataset import AnimalClefDataset
from src.inference import run_inference
import config
import torch

transform = transforms.Compose([
    transforms.Resize([384, 384]),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])

metadata = pd.read_csv(config.METADATA_PATH)
database_df = metadata[metadata['split'] == 'database']
query_df = metadata[metadata['split'] == 'query']

database_dataset = AnimalClefDataset(database_df, root_dir=config.ROOT_DIR, transform=transform)
query_dataset = AnimalClefDataset(query_df, root_dir=config.ROOT_DIR, transform=transform)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
run_inference(database_dataset, query_dataset, device)