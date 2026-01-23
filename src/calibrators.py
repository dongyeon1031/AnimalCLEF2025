from typing import Any, Dict, Optional
import numpy as np
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