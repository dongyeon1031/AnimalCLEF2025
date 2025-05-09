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


def build_aliked(transform, device='cuda', batch_size=16):
    return SimilarityPipeline(
        matcher=MatchLightGlue(features='aliked', device=device, batch_size=batch_size),
        extractor=AlikedExtractor(),
        transform=transform,
        calibration=IsotonicCalibration()
    )

# --- CLIP ViT‑L/14 matcher ---------------------------------
from open_clip import create_model_and_transforms

def build_clip(device='cuda', batch_size=16):
    """
    Build a SimilarityPipeline using CLIP ViT‑L/14 visual encoder
    as a global descriptor (DeepFeatures + Cosine).
    The CLIP package already provides the right preprocessing transform.
    """
    model, _, preprocess = create_model_and_transforms('ViT-L-14', pretrained='openai')
    model = model.visual  # image branch
    model = model.to(device).eval()

    return SimilarityPipeline(
        matcher=CosineSimilarity(),
        extractor=DeepFeatures(model=model, device=device, batch_size=batch_size),
        transform=preprocess,          # PIL -> tensor transform supplied by open_clip
        calibration=IsotonicCalibration()
    )