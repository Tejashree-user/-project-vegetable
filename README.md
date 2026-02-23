# GreenClassify — Deep Learning Vegetable Classifier

This project trains a CNN to classify vegetable images and provides a simple Flask API for inference.

Quick start

1. Create and activate a Python environment (Windows PowerShell example):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Prepare data layout (already present):

```
train/<class>/*
validation/<class>/*
test/<class>/*
```

3. Train:

```powershell
python src/train.py --data_dir . --epochs 10 --batch_size 32
```

4. Run the API:

```powershell
python src/app.py
# then POST an image to http://127.0.0.1:5000/predict
```

Files

- `src/train.py`: training script using transfer learning
- `src/preprocess.py`: EDA and optional resizing
- `src/evaluate.py`: evaluation on `test/`
- `src/app.py`: Flask inference app
- `requirements.txt`: Python dependencies

License: MIT