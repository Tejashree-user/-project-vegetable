import os
from PIL import Image


def class_counts(root):
    for split in ('train','validation','test'):
        path = os.path.join(root, split)
        if not os.path.isdir(path):
            continue
        counts = {}
        for cls in os.listdir(path):
            cls_path = os.path.join(path, cls)
            if os.path.isdir(cls_path):
                counts[cls] = len([f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path,f))])
        print(f"{split} counts:")
        for k,v in sorted(counts.items()):
            print(f"  {k}: {v}")


def resize_images(root, size=(224,224), inplace=False):
    for split in ('train','validation','test'):
        path = os.path.join(root, split)
        if not os.path.isdir(path):
            continue
        for cls in os.listdir(path):
            cls_path = os.path.join(path, cls)
            if not os.path.isdir(cls_path):
                continue
            for fn in os.listdir(cls_path):
                fp = os.path.join(cls_path, fn)
                try:
                    im = Image.open(fp).convert('RGB')
                except Exception:
                    continue
                im = im.resize(size, Image.BILINEAR)
                if inplace:
                    im.save(fp)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='.')
    parser.add_argument('--resize', action='store_true')
    parser.add_argument('--inplace', action='store_true')
    args = parser.parse_args()
    class_counts(args.data_dir)
    if args.resize:
        resize_images(args.data_dir, inplace=args.inplace)
