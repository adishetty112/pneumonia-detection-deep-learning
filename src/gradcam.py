import os,sys,torch,numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.model import build_model
from src.preprocess import inference_transform

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_gradcam(image_path,output_path="results/gradcam.jpg"):
    model=build_model(2).to(DEVICE)
    ckpt=torch.load("models/pneumonia_efficientnet.pth",map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    image=Image.open(image_path).convert("RGB").resize((224,224))
    rgb=np.asarray(image,dtype=np.float32)/255.0
    tensor=inference_transform()(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        cls=int(model(tensor).argmax(1).item())
    target_layers=[model.features[-1]]
    with GradCAM(model=model,target_layers=target_layers) as cam:
        grayscale=cam(input_tensor=tensor,targets=[ClassifierOutputTarget(cls)])[0]
    overlay=show_cam_on_image(rgb,grayscale,use_rgb=True)
    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    Image.fromarray(overlay).save(output_path)
    print("Grad-CAM saved:",output_path)

if __name__=="__main__":
    if len(sys.argv)<2:
        print("Usage: python src/gradcam.py path_to_xray.jpg"); raise SystemExit(1)
    make_gradcam(sys.argv[1])
