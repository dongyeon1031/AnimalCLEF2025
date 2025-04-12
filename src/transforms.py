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