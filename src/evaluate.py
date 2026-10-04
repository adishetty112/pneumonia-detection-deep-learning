import os,sys,torch
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.data_loader import get_loaders
from src.model import build_model

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
_,test_ds,_,test_dl=get_loaders("data/raw/chest_xray",32)
model=build_model(2).to(DEVICE)
ckpt=torch.load("models/pneumonia_efficientnet.pth",map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
p=[]; y=[]
with torch.no_grad():
    for x,t in test_dl:
        p.extend(model(x.to(DEVICE)).argmax(1).cpu().tolist()); y.extend(t.tolist())
print("Accuracy:",accuracy_score(y,p))
print(classification_report(y,p,target_names=test_ds.classes,digits=4))
print("Confusion matrix:\n",confusion_matrix(y,p))
