"""
Tiled ("sliced") inference for dense scenes.

A pile of hundreds of apples filling the frame looks nothing like the
COCO/training images the model learned "apple" from at whole-frame scale.
The fix: cut the frame into overlapping tiles roughly the size of a
single apple's neighborhood, run detection on each tile separately, then
merge all the boxes back into full-frame coordinates.

Merging happens in two passes:
1. IoU-based NMS - removes near-duplicate boxes with high overlap.
2. Distance-based dedupe - removes duplicates that DON'T have high IoU
   (this happens a lot with tiling: the same apple gets detected by two
   neighboring tiles with slightly different crop boundaries, producing
   two boxes of different size/shape for the same apple that don't
   overlap enough for standard IoU-NMS to catch). Without this second
   pass you get exactly what shows up as "extra" boxes/IDs on the same
   apple.
"""

import math
import numpy as np


def _iou(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
    inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def _center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _size(box):
    x1, y1, x2, y2 = box
    return max(x2 - x1, y2 - y1)


def _nms(boxes, confs, iou_threshold=0.35):
    if len(boxes) == 0:
        return []
    idxs = np.argsort(confs)[::-1]
    keep = []
    while len(idxs) > 0:
        current = idxs[0]
        keep.append(current)
        rest = idxs[1:]
        idxs = np.array([i for i in rest if _iou(boxes[current], boxes[i]) < iou_threshold])
    return keep


def _dedupe_by_distance(boxes, confs, keep_idxs, dist_ratio=0.6):
    """Second cleanup pass: merge boxes whose centers are close relative
    to their size, even if their IoU was too low for standard NMS to
    catch (common at tile boundaries)."""
    ordered = sorted(keep_idxs, key=lambda i: confs[i], reverse=True)
    final = []
    final_boxes = []
    for i in ordered:
        c = _center(boxes[i])
        s = _size(boxes[i])
        is_dup = False
        for fb in final_boxes:
            fc = _center(fb)
            fs = _size(fb)
            d = math.hypot(c[0] - fc[0], c[1] - fc[1])
            if d < dist_ratio * max(s, fs):
                is_dup = True
                break
        if not is_dup:
            final.append(i)
            final_boxes.append(boxes[i])
    return final


def _min_area_filter(boxes, confs, min_side=18):
    """Drop tiny noise detections (usually spurious background patches)."""
    return [i for i in range(len(boxes)) if _size(boxes[i]) >= min_side]


def tiled_predict(model, frame, class_ids, conf=0.4, imgsz=640, tile_size=640,
                   overlap=0.3, merge_iou=0.35, dist_ratio=0.6, min_side=18):
    """
    Run detection on overlapping tiles of a frame and merge results.
    class_ids: list of class ids to detect (e.g. [0, 1] for apple + damaged_apple)
    Returns (boxes, confs, classes): boxes is a list of [x1, y1, x2, y2] in
    full-frame pixel coordinates, confs and classes are matching lists
    (classes holds the integer class id for each box).
    """
    h, w = frame.shape[:2]
    step = max(1, int(tile_size * (1 - overlap)))

    all_boxes, all_confs, all_classes = [], [], []

    y = 0
    while y < h:
        y1 = min(y, max(0, h - tile_size))
        y2 = min(y1 + tile_size, h)
        x = 0
        while x < w:
            x1 = min(x, max(0, w - tile_size))
            x2 = min(x1 + tile_size, w)

            tile = frame[y1:y2, x1:x2]
            results = model.predict(tile, classes=class_ids, conf=conf, imgsz=imgsz, verbose=False)

            for r in results:
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                clss = r.boxes.cls.cpu().numpy().astype(int)
                for (bx1, by1, bx2, by2), c, cls_id in zip(xyxy, confs, clss):
                    all_boxes.append([bx1 + x1, by1 + y1, bx2 + x1, by2 + y1])
                    all_confs.append(float(c))
                    all_classes.append(int(cls_id))

            if x2 >= w:
                break
            x += step
        if y2 >= h:
            break
        y += step

    if not all_boxes:
        return [], [], []

    # Pass 1: standard IoU NMS
    keep = _nms(all_boxes, all_confs, merge_iou)
    # Pass 2: distance-based dedupe (catches tile-boundary duplicates IoU misses)
    keep = _dedupe_by_distance(all_boxes, all_confs, keep, dist_ratio)
    # Pass 3: drop tiny noise boxes
    size_ok = set(_min_area_filter([all_boxes[i] for i in keep], [all_confs[i] for i in keep], min_side))
    keep = [k for j, k in enumerate(keep) if j in size_ok]

    return ([all_boxes[i] for i in keep],
            [all_confs[i] for i in keep],
            [all_classes[i] for i in keep])