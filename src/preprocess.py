from torchvision import transforms
from .data_loader import IMAGE_SIZE,MEAN,STD

def inference_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN,STD)
    ])
