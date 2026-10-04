# Pneumonia Detection using Deep Learning

Binary classification of chest X-rays into **NORMAL** and **PNEUMONIA** using EfficientNet-B0 transfer learning.

## Pipeline
Chest X-ray -> preprocessing -> EfficientNet-B0 -> prediction -> Grad-CAM explanation.

## Run
1. Put the dataset at `data/raw/chest_xray`.
2. Activate the virtual environment.
3. Train: `python src/train.py`
4. Evaluate: `python src/evaluate.py`
5. Demo: `streamlit run app/app.py`

The dataset and trained model are not stored in this repository.
