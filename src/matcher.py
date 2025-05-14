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

# --- CLIP ViT‑L/14 matcher ---------------------------------
try:
    from open_clip import create_model_and_transforms
except ImportError:
    create_model_and_transforms = None  # handled in build_clip

def build_clip(device='cuda', batch_size=16, transform=None):
    """
    Build a SimilarityPipeline using CLIP ViT‑L/14 image encoder
    (OpenAI weights) as a global descriptor.
    Requires `open_clip_torch` package.
    """
    if create_model_and_transforms is None:
        raise ImportError("open_clip_torch is not installed. "
                          "Install with `pip install open_clip_torch`.")

    model, _, preprocess = create_model_and_transforms(
        'EVA02-L-14-336',         # valid 336‑px EVA02 model
        pretrained='merged2b_s6b_b61k'   # available weight tag
    )
    model = model.visual.to(device).eval()

    return SimilarityPipeline(
        matcher=CosineSimilarity(),
        extractor=DeepFeatures(model=model, device=device, batch_size=batch_size),
        transform=(transform or preprocess),
        calibration=IsotonicCalibration()
    )