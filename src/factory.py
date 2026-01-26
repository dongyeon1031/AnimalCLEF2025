from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.similarity.pairwise.lightglue import MatchLightGlue
from wildlife_tools.features import DeepFeatures
from wildlife_tools.features.local import AlikedExtractor

from .base import UniversalPipeline
from .scorers import FeatureBasedScorer
from .calibrators import IsotonicCalibrator
from config import EVA_NAME, EVA_WEIGHT_NAME

# ---------------------------------------------------------
# 1. MegaDescriptor 빌더 (Global Feature)
# ---------------------------------------------------------
def build_megadescriptor(model, transform, device='cuda', batch_size=16):
    # Scorer: 특징 추출(DeepFeatures) + 코사인 유사도
    scorer = FeatureBasedScorer(
        extractor=DeepFeatures(model=model, device=device, batch_size=batch_size),
        similarity_metric=CosineSimilarity()
    )
    
    # Calibrator: Isotonic
    calibrator = IsotonicCalibrator()
    
    # 파이프라인 조립
    pipeline = UniversalPipeline(scorer, calibrator)
    
    # 전처리는 파이프라인 외부(데이터셋 로드 시) 혹은 
    # extractor 내부에서 처리되므로 여기선 metadata 처리를 위해 기록만 해둘 수도 있음
    # (DeepFeatures는 transform을 입력으로 받지 않고, 데이터셋의 transform을 사용함)
    return pipeline


# ---------------------------------------------------------
# 2. ALIKED 빌더 (Local Feature)
# ---------------------------------------------------------
def build_aliked(transform=None, device='cuda', batch_size=16):
    """
    ALIKED도 FeatureBasedScorer 구조에 딱 맞습니다.
    Extractor: AlikedExtractor (키포인트 추출)
    Similarity: MatchLightGlue (키포인트 매칭)
    """
    scorer = FeatureBasedScorer(
        extractor=AlikedExtractor(device=device),  # 내부적으로 배치 처리
        similarity_metric=MatchLightGlue(features='aliked', device=device, batch_size=batch_size)
    )
    
    calibrator = IsotonicCalibrator()
    return UniversalPipeline(scorer, calibrator)


# ---------------------------------------------------------
# 3. EVA02 빌더 (CLIP based Global)
# ---------------------------------------------------------
def build_eva02(device='cuda', batch_size=64):
    try:
        from open_clip import create_model_and_transforms
    except ImportError:
        raise ImportError("open_clip_torch not installed.")

    print(f"[Factory] Loading EVA02 model: {EVA_NAME}...")
    model, _, preprocess = create_model_and_transforms(
        EVA_NAME,
        pretrained=EVA_WEIGHT_NAME
    )
    model = model.visual.to(device).eval()

    scorer = FeatureBasedScorer(
            extractor=DeepFeatures(model=model, device=device, batch_size=batch_size),
            similarity_metric=CosineSimilarity()
        )
    
    pipeline = UniversalPipeline(scorer, IsotonicCalibrator())
    pipeline.transform = preprocess
    
    return pipeline