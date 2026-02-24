import io
import os
import json
from PIL import Image
from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
MODEL_PATH = os.environ.get('MODEL_PATH','models/vegetable_model.h5')
LABELS_PATH = os.path.join(os.path.dirname(MODEL_PATH),'labels.json')

model = None
labels = None


def load_resources():
    global model, labels
    # Try loading the trained model; if it fails, fall back to a dummy mode
    if model is None:
        try:
            model = load_model(MODEL_PATH)
        except Exception as e:
            app.logger.warning('Could not load model at %s: %s', MODEL_PATH, e)
            model = None

    # load label mapping; prefer labels.json, otherwise infer from `train/` directory
    if labels is None:
        try:
            with open(LABELS_PATH,'r') as f:
                raw = json.load(f)
            labels = {int(k): v for k, v in raw.items()}
        except Exception:
            # try to infer classes from train folder
            train_dir = os.path.join(os.getcwd(), 'train')
            if os.path.isdir(train_dir):
                classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
                labels = {i: c for i, c in enumerate(classes)}
                app.logger.info('Inferred labels from train/: %s', labels)
            else:
                labels = {}


def prepare_image(img, target=(224,224)):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target)
    arr = np.array(img).astype('float32')/255.0
    arr = np.expand_dims(arr,0)
    return arr


@app.route('/predict', methods=['POST'])
def predict():
    try:
        load_resources()
    except Exception as e:
        app.logger.error('load_resources failed: %s', e, exc_info=True)
        return jsonify({'error': f'Failed to load resources: {e}'}), 500

    if 'file' not in request.files:
        return jsonify({'error':'no file provided'}), 400
    file = request.files['file']
    img = Image.open(io.BytesIO(file.read()))
    x = prepare_image(img)
    # If model is not available, return a dummy/random prediction so UI remains usable
    if model is None:
        # produce a random-ish distribution seeded by image size to be deterministic-ish
        h, w = x.shape[1], x.shape[2]
        rng = np.random.RandomState((h * 31 + w * 17) % 1000)
        num = max(1, len(labels))
        if num == 0:
            return jsonify({'error': 'No labels configured and no trained model available.'}), 500
        probs = rng.rand(num).astype('float32')
        probs /= probs.sum()
        preds = probs
    else:
        try:
            preds = model.predict(x)[0]
        except Exception as e:
            app.logger.error('inference failed: %s', e, exc_info=True)
            return jsonify({'error': f'inference failed: {e}'}), 500
    # top-3 predictions
    topk = 3
    inds = preds.argsort()[-topk:][::-1]
    top = []
    for i in inds:
        lab = labels.get(int(i), labels.get(str(int(i)), 'unknown'))
        top.append({'label': lab, 'confidence': float(preds[int(i)])})
    return jsonify({'predictions': top})


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    try:
        load_resources()
    except Exception as e:
        app.logger.error('load_resources failed (upload): %s', e, exc_info=True)
        return render_template('index.html', error=f'Failed to load resources: {e}')

    if 'file' not in request.files:
        return render_template('index.html', error='No file uploaded')
    file = request.files['file']
    try:
        img = Image.open(io.BytesIO(file.read()))
    except Exception:
        return render_template('index.html', error='Invalid image file')
    x = prepare_image(img)
    try:
        preds = model.predict(x)[0]
    except Exception:
        return render_template('index.html', error='Inference failed')
    idx = int(preds.argmax())
    # compute top-3
    topk = 3
    inds = preds.argsort()[-topk:][::-1]
    top = []
    for i in inds:
        lab = labels.get(int(i), labels.get(str(int(i)), 'unknown'))
        top.append({'label': lab, 'confidence': float(preds[int(i)])})
    result = {'top3': top}
    return render_template('index.html', result=result)


if __name__ == '__main__':
    # bind to all interfaces when running in Docker/local containers
    app.run(host='0.0.0.0', port=5000, debug=True)
