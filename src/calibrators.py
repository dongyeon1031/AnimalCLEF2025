from typing import Any, Dict, Optional
import numpy as np
import xgboost as xgb
from .base import BaseCalibrator
from wildlife_tools.similarity.calibration import IsotonicCalibration as WildlifeIsotonic

class IsotonicCalibrator(BaseCalibrator):
    """
    Isotonic Regression 캘리브레이터 인터페이스
    """
    def __init__(self):
        super().__init__()
        self.model = WildlifeIsotonic()

    def fit(
        self, 
        scores: np.ndarray, 
        labels: np.ndarray, 
        metadata: Optional[Dict[str, Any]] = None
    ):
        # Isotonic은 metadata를 쓰지 않으므로 무시하고 점수만 넘김
        self.model.fit(scores, labels)
        self.is_fitted = True

    def predict(
        self, 
        scores: np.ndarray, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        return self.model.predict(scores)


class XGBoostCalibrator(BaseCalibrator):
    """
    XGBoost 기반 캘리브레이터
    """
    def __init__(self, **params):
        super().__init__()
        # 기본 하이퍼파라미터 설정
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric='logloss',
            **params
        )

    def fit(self, scores, labels, metadata=None):
        # XGBoost는 2차원 배열(N, Feature_size)을 입력으로 받으므로 형태 변환
        X = scores.reshape(-1, 1)
        y = labels
        
        print("[XGBoost] Fitting model with calibration data...")
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, scores, metadata=None):
        if not self.is_fitted:
            print("[Warning] XGBoostCalibrator is not fitted yet!")
            return scores
            
        X = scores.reshape(-1, 1)
        # 클래스 1(Match)에 대한 확률값만 추출
        probs = self.model.predict_proba(X)[:, 1]
        return probs
