from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.similarity.pairwise.lightglue import MatchLightGlue
from wildlife_tools.similarity.wildfusion import SimilarityPipeline
from wildlife_tools.features import DeepFeatures
from wildlife_tools.features.local import AlikedExtractor
from wildlife_tools.similarity.calibration import IsotonicCalibration

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
        'EVA02-L-14-336',
        pretrained='merged2b_s6b_b61k'
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