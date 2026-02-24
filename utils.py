import os
import json

def save_labels_from_train(train_dir, out_dir='models'):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    gen = ImageDataGenerator(rescale=1./255)
    flow = gen.flow_from_directory(train_dir, target_size=(224,224), batch_size=4, class_mode='categorical')
    mapping = {v:k for k,v in flow.class_indices.items()}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir,'labels.json'),'w') as f:
        json.dump(mapping,f)
    print('Saved labels.json')

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--train_dir', default='train', help='path to training directory')
    args = p.parse_args()
    save_labels_from_train(args.train_dir)
