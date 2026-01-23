from typing import Any, Callable, List, Optional, Tuple
import numpy as np
from .base import BaseScorer

class FeatureBasedScorer(BaseScorer):
    """
    [Scorer 구현체 1] 특징 기반 점수 계산기
    (MegaDescriptor, EVA02, DINOv3 등 Global Descriptor용)
    
    작동 과정:
    1. 데이터셋 전체에서 특징 벡터(Embedding) 추출
    2. 추출된 벡터들 간의 유사도(코사인 등)를 행렬 연산으로 한 번에 계산
    """
    def __init__(
        self, 
        extractor: Callable, 
        similarity_metric: Callable
    ):
        """
        Args:
            extractor: 데이터셋 -> 특징 행렬(Features) 변환 함수 
                       (예: wildlife_tools.features.DeepFeatures)
            similarity_metric: 특징 행렬 2개 -> 유사도 행렬 변환 함수 
                               (예: wildlife_tools.similarity.CosineSimilarity)
        """
        self.extractor = extractor
        self.similarity_metric = similarity_metric

    def compute_score_matrix(
        self, 
        query_dataset: Any, 
        db_dataset: Any, 
        pairs: Optional[List[Any]] = None
    ) -> np.ndarray:
        
        # 1. 특징 추출 (Feature Extraction)
        # extractor가 내부적으로 (N, D) 크기의 벡터 행렬을 반환한다고 가정
        feat_query = self.extractor(query_dataset)
        feat_db = self.extractor(db_dataset)

        # 2. 유사도 계산 (Similarity Calculation)
        if pairs is None:
            # 전체 N x M 행렬 계산
            return self.similarity_metric(feat_query, feat_db)
        else:
            # wildlife_tools의 CosineSimilarity 등은 보통 전체 행렬을 반환하므로
            # 전체를 계산한 뒤 필요한 pair만 인덱싱하는 방식을 사용 (구현에 따라 최적화 가능)
            full_matrix = self.similarity_metric(feat_query, feat_db)
            
            # pairs = [(q_idx, db_idx), ...] 형태라고 가정
            pairs_arr = np.array(pairs)
            q_indices = pairs_arr[:, 0]
            db_indices = pairs_arr[:, 1]
            
            # 해당 위치의 점수만 1D Array로 반환
            return full_matrix[q_indices, db_indices]


class PairwiseScorer(BaseScorer):
    """
    [Scorer 구현체 2] 쌍(Pair) 기반 점수 계산기
    (LoFTR, RoMa, LightGlue 등 Local Feature Matcher용)
    
    작동 원리:
    1. 특징을 미리 추출하지 않고(또는 못하고), 
    2. 쿼리와 DB 이미지를 쌍으로 묶어 모델에 직접 넣어 점수를 계산합니다.
    """
    def __init__(self, model_inference_fn: Callable):
        """
        Args:
            model_inference_fn: 두 이미지(또는 데이터)를 받아 점수를 리턴하는 함수
                                def func(img_q, img_db) -> float
        """
        self.model_inference_fn = model_inference_fn

    def compute_score_matrix(
        self, 
        query_dataset: Any, 
        db_dataset: Any, 
        pairs: Optional[List[Any]] = None
    ) -> np.ndarray:
        
        # 1. Pairs가 지정되지 않은 경우: 모든 조합(N x M) 계산
        # (주의: LoFTR 등은 느리므로 데이터가 많을 땐 pairs 없이 호출하면 매우 오래 걸릴 수 있음)
        if pairs is None:
            n_query = len(query_dataset)
            n_db = len(db_dataset)
            scores = np.zeros((n_query, n_db), dtype=np.float32)
            
            print(f"[PairwiseScorer] Computing all {n_query * n_db} pairs...")
            for i in range(n_query):
                q_data = query_dataset[i]
                for j in range(n_db):
                    db_data = db_dataset[j]
                    # 모델 추론
                    scores[i, j] = self.model_inference_fn(q_data, db_data)
            return scores
        
        # 2. Pairs가 지정된 경우: 해당 목록만 계산 (효율적)
        else:
            scores = []
            for q_idx, db_idx in pairs:
                q_data = query_dataset[q_idx]
                db_data = db_dataset[db_idx]
                score = self.model_inference_fn(q_data, db_data)
                scores.append(score)
            return np.array(scores)