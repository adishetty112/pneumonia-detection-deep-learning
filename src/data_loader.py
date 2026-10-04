import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

IMAGE_SIZE=224
MEAN=[0.485,0.456,0.406]
STD=[0.229,0.224,0.225]

def get_loaders(data_dir="data/raw/chest_xray", batch_size=32):
    train_tf=transforms.Compose([
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ToTensor(),
        transforms.Normalize(MEAN,STD)
    ])
    test_tf=transforms.Compose([
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN,STD)
    ])
    train_ds=datasets.ImageFolder(os.path.join(data_dir,"train"),transform=train_tf)
    test_ds=datasets.ImageFolder(os.path.join(data_dir,"test"),transform=test_tf)
    train_dl=DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=0)
    test_dl=DataLoader(test_ds,batch_size=batch_size,shuffle=False,num_workers=0)
    return train_ds,test_ds,train_dl,test_dl
