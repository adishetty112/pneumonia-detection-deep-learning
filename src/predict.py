import os,sys,torch
from PIL import Image
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.model import build_model
from src.preprocess import inference_transform

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES=["NORMAL","PNEUMONIA"]

def predict(image_path):
    model=build_model(2).to(DEVICE)
    ckpt=torch.load("models/pneumonia_efficientnet.pth",map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    image=Image.open(image_path).convert("RGB")
    x=inference_transform()(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs=torch.softmax(model(x),dim=1)[0]
    idx=int(probs.argmax())
    return CLASSES[idx],float(probs[idx])

if __name__=="__main__":
    if len(sys.argv)<2:
        print("Usage: python src/predict.py path_to_xray.jpg"); raise SystemExit(1)
    label,confidence=predict(sys.argv[1])
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence*100:.2f}%")
