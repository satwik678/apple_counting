import argparse
import os
import time
from collections import deque

import cv2
from ultralytics import YOLO

from utils.config import load_line
from utils.crossing import LineCrossingCounter
from utils.drawing import TrailDrawer, draw_counting_line
from utils.hud import draw_detection_box, draw_hud
from utils.tiling import generate_tiles, merge_detections
from utils.tracker import CentroidTracker


def get_args():
    parser = argparse.ArgumentParser(
        description="Individual apple detection, tracking and line-crossing counting"
    )

    parser.add_argument(
        "--source",
        type=str,
        default="videos/input.mp4",
        help="Path to input video"
    )

    parser.add_argument(
        "--weights",
        type=str,
        default="models/best.pt",
        help="Path to trained YOLO model"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.20,
        help="Confidence threshold"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="videos/output_counted.mp4",
        help="Path to save output video"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size"
    )

    parser.add_argument(
        "--tile-size",
        type=int,
        default=640,
        help="Tile size for dense frames"
    )

    parser.add_argument(
        "--overlap",
        type=float,
        default=0.30,
        help="Tile overlap"
    )

    parser.add_argument(
        "--no-tile",
        action="store_true",
        help="Disable tiling"
    )

    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Comma-separated class names"
    )

    parser.add_argument(
        "--line-y",
        type=float,
        default=0.55,
        help="Default counting line position"
    )

    parser.add_argument(
        "--line-tilt",
        type=float,
        default=0.10,
        help="Default counting line tilt"
    )

    parser.add_argument(
        "--show-conf",
        action="store_true",
        help="Show confidence scores"
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show live preview"
    )

    return parser.parse_args()


def default_line(width, height, line_y=0.55, line_tilt=0.10):
    y = int(height * line_y)

    x1 = int(width * 0.15)
    x2 = int(width * 0.85)

    tilt = int(height * line_tilt)

    y1 = max(0, min(height - 1, y - tilt))
    y2 = max(0, min(height - 1, y + tilt))

    return (x1, y1), (x2, y2)


def resolve_class_ids(model, requested_names):
    name_to_id = {
        name.lower(): class_id
        for class_id, name in model.names.items()
    }

    if requested_names is None:
        return list(model.names.keys())

    class_ids = []

    for name in requested_names.split(","):
        name = name.strip().lower()

        if name in name_to_id:
            class_ids.append(name_to_id[name])
        else:
            print(
                f"Warning: class '{name}' not found. "
                f"Available classes: {list(name_to_id.keys())}"
            )

    return class_ids


def predict_full_frame(
    model,
    frame,
    class_ids,
    confidence,
    image_size
):
    results = model.predict(
        frame,
        classes=class_ids,
        conf=confidence,
        imgsz=image_size,
        verbose=False
    )

    boxes = []
    confs = []
    class_ids_found = []

    for result in results:
        if result.boxes is None:
            continue

        xyxy = result.boxes.xyxy.cpu().numpy()
        result_confs = result.boxes.conf.cpu().numpy()
        result_classes = result.boxes.cls.cpu().numpy().astype(int)

        for box, conf, class_id in zip(
            xyxy,
            result_confs,
            result_classes
        ):
            boxes.append(box.tolist())
            confs.append(float(conf))
            class_ids_found.append(int(class_id))

    return boxes, confs, class_ids_found


def predict_tiled(
    model,
    frame,
    class_ids,
    confidence,
    image_size,
    tile_size,
    overlap
):
    height, width = frame.shape[:2]

    tiles = generate_tiles(
        width,
        height,
        tile_size,
        overlap
    )

    detections = []

    for tx1, ty1, tx2, ty2 in tiles:

        tile = frame[ty1:ty2, tx1:tx2]

        results = model.predict(
            tile,
            classes=class_ids,
            conf=confidence,
            imgsz=image_size,
            verbose=False
        )

        for result in results:
            if result.boxes is None:
                continue

            xyxy = result.boxes.xyxy.cpu().numpy()
            result_confs = result.boxes.conf.cpu().numpy()
            result_classes = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf, class_id in zip(
                xyxy,
                result_confs,
                result_classes
            ):

                x1, y1, x2, y2 = box

                full_box = [
                    float(x1 + tx1),
                    float(y1 + ty1),
                    float(x2 + tx1),
                    float(y2 + ty1)
                ]

                detections.append(
                    {
                        "box": tuple(full_box),
                        "class_id": int(class_id),
                        "confidence": float(conf)
                    }
                )

    merged = merge_detections(
        detections,
        iou_threshold=0.45
    )

    boxes = [
        list(item["box"])
        for item in merged
    ]

    confs = [
        item["confidence"]
        for item in merged
    ]

    class_ids_found = [
        item["class_id"]
        for item in merged
    ]

    return boxes, confs, class_ids_found


def main():
    args = get_args()

    if not os.path.exists(args.source):
        raise FileNotFoundError(
            f"Could not find input video: {args.source}"
        )

    if not os.path.exists(args.weights):
        raise FileNotFoundError(
            f"Could not find model weights: {args.weights}"
        )

    os.makedirs(
        os.path.dirname(args.output) or ".",
        exist_ok=True
    )

    print("Loading model...")

    model = YOLO(args.weights)

    print("Model classes:")
    print(model.names)

    class_ids = resolve_class_ids(
        model,
        args.classes
    )

    if not class_ids:
        raise ValueError(
            "No valid classes selected."
        )

    capture = cv2.VideoCapture(
        args.source
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {args.source}"
        )

    fps_input = capture.get(
        cv2.CAP_PROP_FPS
    )

    if fps_input <= 0:
        fps_input = 30.0

    width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    print(
        f"Video size: {width} x {height}"
    )

    print(
        f"Input FPS: {fps_input:.2f}"
    )

    counting_line = load_line()

    if counting_line is None:
        counting_line = default_line(
            width,
            height,
            args.line_y,
            args.line_tilt
        )
        print(
            f"Using default counting line: {counting_line}"
        )
    else:
        print(
            f"Using calibrated counting line: {counting_line}"
        )

    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps_input,
        (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not create output video: {args.output}"
        )

    tracker = CentroidTracker(
        max_distance=55,
        max_missed_frames=15
    )

    counter = LineCrossingCounter(
        counting_line[0],
        counting_line[1],
        direction=1
    )

    trail_drawer = TrailDrawer(
        max_length=15
    )

    count_history = deque(
        maxlen=100
    )

    frame_idx = 0
    start_time = time.time()

    print("Starting detection...")
    print("Press Q to stop the video.")

    while True:

        success, frame = capture.read()

        if not success:
            break

        frame_idx += 1

        frame_start = time.time()

        if args.no_tile:

            boxes, confs, detected_class_ids = predict_full_frame(
                model,
                frame,
                class_ids,
                args.conf,
                args.imgsz
            )

        else:

            boxes, confs, detected_class_ids = predict_tiled(
                model,
                frame,
                class_ids,
                args.conf,
                args.imgsz,
                args.tile_size,
                args.overlap
            )

        detected_classes = [
            model.names[class_id]
            for class_id in detected_class_ids
        ]

        tracks = tracker.update(
            boxes,
            detected_classes,
            confs
        )

        for (
            track_id,
            box,
            class_name,
            confidence,
            was_detected
        ) in tracks:

            x1, y1, x2, y2 = box

            center = (
                (x1 + x2) / 2.0,
                (y1 + y2) / 2.0
            )

            trail_drawer.update(
                track_id,
                center
            )

            counter.update(
                track_id,
                center,
                class_name
            )

            draw_detection_box(
                frame,
                box,
                class_name,
                confidence,
                track_id=track_id,
                show_conf=args.show_conf
            )

            cv2.circle(
                frame,
                (
                    int(center[0]),
                    int(center[1])
                ),
                4,
                (255, 255, 255),
                -1
            )

        trail_drawer.draw(
            frame
        )

        draw_counting_line(
            frame,
            counting_line[0],
            counting_line[1]
        )

        visible_count = len(tracks)

        average_confidence = (
            sum(confs) / len(confs)
            if confs
            else 0.0
        )

        elapsed = time.time() - start_time

        processing_fps = (
            frame_idx / elapsed
            if elapsed > 0
            else 0.0
        )

        count_history.append(
            counter.total_count
        )

        draw_hud(
            frame,
            counter.get_counts(),
            counter.total_count,
            visible_count,
            average_confidence,
            processing_fps,
            list(count_history)
        )

        writer.write(
            frame
        )

        if args.show:

            cv2.imshow(
                "Apple Counting",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

        frame_time = time.time() - frame_start

        if frame_idx % 30 == 0:

            print(
                f"Frame {frame_idx} | "
                f"Crossed: {counter.total_count} | "
                f"Tracked: {visible_count} | "
                f"{frame_time:.2f}s/frame"
            )

    capture.release()
    writer.release()
    cv2.destroyAllWindows()

    print()
    print("Detection completed.")
    print(
        f"Final class counts: {counter.get_counts()}"
    )
    print(
        f"Total counted: {counter.total_count}"
    )
    print(
        f"Output saved to: {args.output}"
    )


if __name__ == "__main__":
    main()