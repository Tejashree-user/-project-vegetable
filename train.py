import argparse
import os
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam


def build_model(num_classes, weights='imagenet'):
    # allow running without pretrained weights for quick smoke tests
    w = None if (weights is None or str(weights).lower() in ('none','null','no')) else 'imagenet'
    base = MobileNetV2(weights=w, include_top=False, input_shape=(224,224,3))
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(base.input, out)
    for layer in base.layers:
        layer.trainable = False
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='.', help='root folder containing train/validation/test')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weights', default='imagenet', help="'imagenet' or 'none' to skip pretrained weights")
    parser.add_argument('--out', default='models/vegetable_model.h5')
    args = parser.parse_args()

    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'validation')

    train_aug = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.05,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_aug = ImageDataGenerator(rescale=1./255)

    train_gen = train_aug.flow_from_directory(train_dir, target_size=(224,224), batch_size=args.batch_size, class_mode='categorical')
    val_gen = val_aug.flow_from_directory(val_dir, target_size=(224,224), batch_size=args.batch_size, class_mode='categorical')

    num_classes = len(train_gen.class_indices)
    model = build_model(num_classes, weights=args.weights)
    model.compile(optimizer=Adam(args.lr), loss='categorical_crossentropy', metrics=['accuracy'])

    try:
        model.fit(train_gen, validation_data=val_gen, epochs=args.epochs)
    except KeyboardInterrupt:
        print('Training interrupted by user')
    except Exception as e:
        print('Training failed:', e)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model.save(args.out)

    mapping = {v:k for k,v in train_gen.class_indices.items()}
    with open(os.path.join(os.path.dirname(args.out), 'labels.json'), 'w') as f:
        json.dump(mapping, f)

    print('Saved model to', args.out)


if __name__ == '__main__':
    main()
