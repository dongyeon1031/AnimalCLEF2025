from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Callable
import numpy as np

class BaseScorer(ABC):
    """
    [추상 클래스] 점수 계산기
    어떤 방식(Global Descriptor, Local Feature, Pairwise Model 등)이든
    두 데이터셋 간의 유사도 점수 행렬을 계산하는 인터페이스를 통일합니다.
    """

    @abstractmethod
    def compute_score_matrix(
        self, 
        query_dataset: Any, 
        db_dataset: Any,
        pairs: Optional[List[Any]] = None
    ) -> np.ndarray:
        """
        Query와 DB 데이터셋을 입력받아 유사도 점수(Score)를 반환합니다.

        Args:
            query_dataset: 쿼리 이미지 데이터셋
            db_dataset: 데이터베이스 이미지 데이터셋
            pairs: (Optional) 특정 쌍(pair)만 계산하고 싶을 때 [(q_idx, db_idx), ...] 형태의 리스트 제공.
                   None일 경우 전체 행렬(N x M)을 계산함.

        Returns:
            np.ndarray: (N_query, N_db) 형태의 점수 행렬.
                        pairs가 제공된 경우, 해당 위치에만 점수가 채워지거나 
                        구현에 따라 1D array가 반환될 수도 있음(구현체에 따름).
        """
        pass

    def __call__(self, query_dataset, db_dataset, pairs=None):
        return self.compute_score_matrix(query_dataset, db_dataset, pairs=pairs)


class BaseCalibrator(ABC):
    """
    [추상 클래스] 확률 보정기
    Scorer가 뱉은 점수(Score)를 확률(Probability)로 변환합니다.
    기존과 달리 metadata를 입력받아 더 정교한 보정이 가능하도록 설계되었습니다.
    """
    
    def __init__(self):
        self.is_fitted = False

    @abstractmethod
    def fit(
        self, 
        scores: np.ndarray, 
        labels: np.ndarray, 
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        보정 모델을 학습합니다.

        Args:
            scores: (N,) 1D float array (유사도 점수)
            labels: (N,) 1D boolean/int array (정답 여부, 1=Match, 0=Non-match)
            metadata: (Optional) 학습에 사용할 추가 정보 (예: {'img_quality': ...})
        """
        pass

    @abstractmethod
    def predict(
        self, 
        scores: np.ndarray, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """
        점수를 확률(0~1)로 변환합니다.
        """
        pass


class UniversalPipeline:
    """
    [통합 파이프라인]
    Scorer와 Calibrator를 결합하여 최종 Re-ID 시스템을 구성합니다.
    모델 종류에 상관없이 공통된 사용법을 제공합니다.
    """
    
    def __init__(
        self, 
        scorer: BaseScorer, 
        calibrator: Optional[BaseCalibrator] = None
    ):
        self.scorer = scorer
        self.calibrator = calibrator

    def _compute_hits(self, dataset0, dataset1):
        """
        두 데이터셋의 라벨을 비교하여 정답지(Ground Truth) 행렬을 생성합니다.
        (True: 같은 개체, False: 다른 개체)
        """
        gt0 = dataset0.labels_string
        gt1 = dataset1.labels_string
        # N x 1 와 1 x M 을 비교하여 N x M 행렬 생성 (Broadcasting)
        gt_grid0 = np.tile(gt0, (len(gt1), 1)).T
        gt_grid1 = np.tile(gt1, (len(gt0), 1))
        return (gt_grid0 == gt_grid1)

    def fit_calibration(
        self, 
        dataset0: Any, 
        dataset1: Any, 
        metadata_extractor: Optional[Callable] = None
    ):
        """
        캘리브레이션 모델을 학습시킵니다.

        Args:
            dataset0: 학습용 데이터셋 1
            dataset1: 학습용 데이터셋 2
            metadata_extractor: (Optional) 함수 f(d0, d1, scores) -> dict
                                캘리브레이션에 필요한 메타데이터를 추출하는 사용자 함수
        """
        if self.calibrator is None:
            print("[Warning] Pipeline has no calibrator. Skipping fit.")
            return

        print(f"[Pipeline] Computing raw scores with {type(self.scorer).__name__}...")
        # 1. Raw Score 계산
        scores = self.scorer(dataset0, dataset1)

        # 2. 정답지(Label) 생성
        hits = self._compute_hits(dataset0, dataset1)

        # 3. 메타데이터 추출 (확장 포인트)
        meta = None
        if metadata_extractor is not None:
            print("[Pipeline] Extracting metadata...")
            meta = metadata_extractor(dataset0, dataset1, scores)

        # 4. 학습 수행
        print("[Pipeline] Fitting calibrator...")
        self.calibrator.fit(scores.flatten(), hits.flatten(), metadata=meta)
        self.calibrator.is_fitted = True
        print("[Pipeline] Calibration fitted successfully.")

    def __call__(
        self, 
        dataset0: Any, 
        dataset1: Any, 
        pairs: Optional[List[Any]] = None,
        metadata_extractor: Optional[Callable] = None
    ) -> np.ndarray:
        """
        최종 유사도(또는 확률) 점수를 계산합니다.
        """
        # 1. Scorer를 통해 점수 계산
        scores = self.scorer(dataset0, dataset1, pairs=pairs)

        # 2. Calibrator가 있고 학습되었다면 확률 보정 적용
        if self.calibrator is not None:
            if not self.calibrator.is_fitted:
                # 학습 안 됐으면 경고 후 Raw Score 반환 (혹은 에러 처리)
                print("[Warning] Calibrator is not fitted. Returning raw scores.")
                return scores

            # 메타데이터 추출
            meta = None
            if metadata_extractor is not None:
                meta = metadata_extractor(dataset0, dataset1, scores)
            
            # 형태 보존하며 predict 호출
            original_shape = scores.shape
            
            # pairs가 있을 경우 1D array일 수 있으므로 상황에 맞게 처리
            flat_scores = scores.flatten()
            
            # 보정 예측
            probs = self.calibrator.predict(flat_scores, metadata=meta)
            
            # 원래 형태로 복구
            scores = probs.reshape(original_shape)

        return scores