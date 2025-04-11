import timm

def load_model(name='hf-hub:BVRA/MegaDescriptor-L-384', device='cuda'):
    model = timm.create_model(name, num_classes=0, pretrained=True)
    model.to(device)
    model.eval()
    return model