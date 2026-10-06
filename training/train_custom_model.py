
"""
Optional: fine-tune YOLO26 on your own labeled apple dataset.

You only need this if the pretrained COCO "apple" class isn't accurate
enough for your setup (e.g. unusual lighting, overlapping fruit, a
specific apple variety/defect you need to distinguish).

USAGE:
    1. Label your data (e.g. with Roboflow or CVAT) in YOLO format.
    2. Create a data.yaml describing your train/val paths and class names.
    3. Run:
       python training/train_custom_model.py --data path/to/data.yaml --base-weights yolo26s.pt --epochs 250

Trained weights are saved under runs/detect/train/weights/best.pt.
Copy that file into the models/ folder and point --weights at it in
detection.py once training is done.
"""

import argparse
from ultralytics import YOLO


def get_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLO26 on a custom apple dataset"
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to your data.yaml file"
    )

    parser.add_argument(
        "--base-weights",
        type=str,
        default="yolo26s.pt",
        help="Starting checkpoint to fine-tune from"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=250
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Stop early if validation performance plateaus"
    )

    return parser.parse_args()


def main():
    args = get_args()

    model = YOLO(args.base_weights)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        patience=args.patience
    )


if __name__ == "__main__":
    main()
