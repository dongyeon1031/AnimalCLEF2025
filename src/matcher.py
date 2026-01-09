from config import EVA_NAME, EVA_WEIGHT_NAME
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.similarity.pairwise.lightglue import MatchLightGlue
from wildlife_tools.similarity.wildfusion import SimilarityPipeline
from wildlife_tools.features import DeepFeatures
from wildlife_tools.features.local import AlikedExtractor
from wildlife_tools.similarity.calibration import IsotonicCalibration
import torch
import torchvision

'''
MegaDescriptor, ALIKED matcher 각각 생성 + return
'''
def build_megadescriptor(model, transform, device='cuda', batch_size=16):
    return SimilarityPipeline(
        matcher=CosineSimilarity(),
        extractor=DeepFeatures(model=model, device=device, batch_size=batch_size),
        transform=transform,
        calibration=IsotonicCalibration()
    )

# --- ALIKED matcher ---------------------------------
def build_aliked(transform, device='cuda', batch_size=16):
    return SimilarityPipeline(
        matcher=MatchLightGlue(features='aliked', device=device, batch_size=batch_size),
        extractor=AlikedExtractor(),
        transform=transform,
        calibration=IsotonicCalibration()
    )

# --- EVA02-CLIP matcher -----------------------------------
try:
    from open_clip import create_model_and_transforms
except ImportError:
    create_model_and_transforms = None

def build_eva02(device='cuda', batch_size=16):
    """
    Global descriptor from EVA02-CLIP-L-14-336.
    Returns SimilarityPipeline with Cosine similarity.
    """
    if create_model_and_transforms is None:
        raise ImportError("open_clip_torch not installed. pip install open_clip_torch")

    model, _, preprocess = create_model_and_transforms(
        EVA_NAME,
        pretrained=EVA_WEIGHT_NAME
    )
    model = model.visual.to(device).eval()

    def eva_transform(img, metadata=None):  # 메타데이터 무시하도록 래핑
        return preprocess(img)

    return SimilarityPipeline(
        matcher=CosineSimilarity(),
        extractor=DeepFeatures(model, device=device, batch_size=batch_size),
        transform=eva_transform,
        calibration=IsotonicCalibration()
    )

# --- DinoV3 matcher -----------------------------------
def build_dinov3(device='cuda', batch_size=16):
    # 1) model load (torch.hub 사용해서 로드)
    model = torch.hub.load("facebookresearch/dinov3", "dinov3_vits14")
    model = model.to(device).eval()

    # 2) DINOv3 output wrapper: dict -> (B, D)
    class DinoV3Embedder(torch.nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone

        def forward(self, x):   # 여기 검증하기
            out = self.backbone(x)
            # DINOv3가 dict를 내는 케이스 대응
            if isinstance(out, dict):
                # 가장 흔한 전역 임베딩 키
                return out["x_norm_clstoken"]
            return out  # 이미 (B, D)면 그대로

    model = DinoV3Embedder(model).to(device).eval()

    # 3) preprocess (이미지넷 스타일 씀)
    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                         std=(0.229, 0.224, 0.225)),
    ])

    def dino_transform(img, metadata=None):
        return preprocess(img)

    return SimilarityPipeline(
        matcher=CosineSimilarity(),
        extractor=DeepFeatures(model=model, device=device, batch_size=batch_size),
        transform=dino_transform,
        calibration=IsotonicCalibration()
    )