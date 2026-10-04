import os, sys, torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.data_loader import get_loaders
from src.model import build_model

DATA_DIR="data/raw/chest_xray"
MODEL_PATH="models/pneumonia_efficientnet.pth"
BATCH_SIZE=32
EPOCHS=3
LR=1e-4
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_ds,test_ds,train_dl,test_dl=get_loaders(DATA_DIR,BATCH_SIZE)
print("Device:",DEVICE)
print("Classes:",train_ds.classes)
print("Train:",len(train_ds),"Test:",len(test_ds))

model=build_model(2).to(DEVICE)
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.classifier[1].parameters(),lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total=correct=0
    loss_sum=0.0
    for x,y in train_dl:
        x,y=x.to(DEVICE),y.to(DEVICE)
        optimizer.zero_grad()
        out=model(x)
        loss=criterion(out,y)
        loss.backward()
        optimizer.step()
        loss_sum+=loss.item()
        correct+=(out.argmax(1)==y).sum().item()
        total+=y.size(0)
    print(f"Epoch {epoch+1}/{EPOCHS} | loss={loss_sum/len(train_dl):.4f} | acc={correct/total:.4f}")

model.eval()
preds=[]; labels=[]
with torch.no_grad():
    for x,y in test_dl:
        out=model(x.to(DEVICE))
        preds.extend(out.argmax(1).cpu().tolist())
        labels.extend(y.tolist())

print("\nAccuracy:",accuracy_score(labels,preds))
print("\nClassification report:\n",classification_report(labels,preds,target_names=test_ds.classes,digits=4))
print("Confusion matrix:\n",confusion_matrix(labels,preds))

os.makedirs("models",exist_ok=True)
torch.save({"model_state_dict":model.state_dict(),"classes":train_ds.classes},MODEL_PATH)
print("\nSaved:",MODEL_PATH)
