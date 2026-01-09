try:
    import torch
    from kornia.feature import LoFTR
except ImportError:
    torch = None
    LoFTR = None

class LoFTRDenseMatcher:
    """
    Dense matching wrapper for LoFTR.
    This class is NOT used inside SimilarityPipeline yet.
    It is intended for future top-K re-ranking with dense correspondence scores.
    """
    def __init__(self, device='cuda'):
        if LoFTR is None:
            raise ImportError("kornia is not installed. Please install kornia to use LoFTRDenseMatcher.")
        self.device = device
        if torch is None:
            raise ImportError("PyTorch is required to use this matcher.")
        # Kornia LoFTR provides pretrained weights via the `pretrained` argument.
        # Use 'outdoor' by default which is typically more suitable for in-the-wild imagery.
        self.model = LoFTR(pretrained='outdoor').to(device).eval()

    def match(self, img_q, img_g):
        """
        img_q, img_g: preprocessed torch.Tensor images (1, 1 or 3, H, W)
        Returns raw correspondence dict from LoFTR.
        Score reduction is intentionally left undefined.
        """
        with torch.no_grad():
            out = self.model(img_q, img_g)
        return out


def build_loftr(device='cuda'):
    """
    Build a LoFTR dense matcher stub.
    This does NOT return a SimilarityPipeline.
    It only prepares the matcher object for future dense re-ranking experiments.
    """
    return LoFTRDenseMatcher(device=device)