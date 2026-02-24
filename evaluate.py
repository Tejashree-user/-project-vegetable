import os
import json
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix


def main(data_dir='.'):
    test_dir = os.path.join(data_dir, 'test')
    model_path = os.path.join('models','vegetable_model.h5')
    labels_path = os.path.join('models','labels.json')

    from tensorflow.keras.models import load_model
    model = load_model(model_path)
    with open(labels_path,'r') as f:
        labels = json.load(f)

    gen = ImageDataGenerator(rescale=1./255)
    test_gen = gen.flow_from_directory(test_dir, target_size=(224,224), batch_size=32, class_mode='categorical', shuffle=False)

    # evaluate on test set (faster without full confusion matrix)
    print('Evaluating model on test set...')
    loss, accuracy = model.evaluate(test_gen, verbose=0)
    print(f'Test Loss: {loss:.4f}')
    print(f'Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)')
    print('\nModel evaluation complete!')

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='.')
    args = p.parse_args()
    main(args.data_dir)
