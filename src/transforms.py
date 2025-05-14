import torch

# ---------- Five‑Crop TTA (average) helper -----------------
class FiveCropAverage:
    """
    Apply torchvision FiveCrop then average the 5 cropped tensors into one.
    Output tensor shape == single image tensor; thus DeepFeatures works unchanged.
    """
    def __init__(self, size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.five_crop = T.FiveCrop(size)
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=mean, std=std)

    def __call__(self, img):
        crops = self.five_crop(img)          # tuple of 5 PIL images
        tensors = [self.normalize(self.to_tensor(c)) for c in crops]
        stacked = torch.stack(tensors, dim=0)  # 5 x C x H x W
        return stacked.mean(dim=0)            # C x H x W (averaged)
import torchvision.transforms as T

# MegaDescriptor / ConvNeXt transforms
transform_display = T.Compose([
    T.Resize([384, 384]),
])
transform = T.Compose([
    *transform_display.transforms,
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225))
])

# ALIKED transforms (no normalize)
transforms_aliked = T.Compose([
    T.Resize([512, 512]),
    T.ToTensor()
])
# Mega 5‑crop TTA transform
transform_tta_mega = T.Compose([
    T.Resize([384, 384]),
    FiveCropAverage(384)
])

# CLIP 336 Five‑crop – we rely on CLIP mean/std (same as ImageNet)
transform_tta_clip = T.Compose([
    T.Resize([336, 336]),
    FiveCropAverage(336)
])