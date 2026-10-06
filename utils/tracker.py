"""
Centroid tracker with position smoothing and gap-bridging.

Two flicker sources this addresses:
1. Frame-to-frame detection noise makes a box jitter slightly even when
   the apple hasn't moved - fixed with EMA (exponential moving average)
   smoothing of each tracked box's position.
2. A single missed detection (common with confidence noise) used to make
   a box vanish for a frame and reappear as a "new" apple next frame -
   fixed by keeping a track alive at its last known (smoothed) position
   for a short grace period, so it coasts through a brief miss instead
   of flickering off.
"""

import math


class CentroidTracker:
    def __init__(self, max_distance=55, max_missed_frames=8,
                 size_ratio_tolerance=1.8, smoothing_alpha=0.55):
        self.next_id = 0
        self.objects = {}       # id -> (cx, cy) smoothed centroid
        self.boxes = {}         # id -> smoothed [x1, y1, x2, y2]
        self.sizes = {}         # id -> box size (max side length)
        self.classes = {}       # id -> last known class name
        self.confs = {}         # id -> last known confidence
        self.missed = {}        # id -> frames since last real detection
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames
        self.size_ratio_tolerance = size_ratio_tolerance
        self.smoothing_alpha = smoothing_alpha   # weight on the NEW detection (higher = snappier, less smooth)
        self.total_ids_ever = 0

    @staticmethod
    def _centroid(box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _size(box):
        x1, y1, x2, y2 = box
        return max(x2 - x1, y2 - y1)

    def update(self, boxes, classes, confs):
        """
        boxes/classes/confs: parallel lists for THIS frame's raw detections.

        Returns a list of (track_id, box, class_name, conf, was_detected_this_frame)
        for every currently active track - both freshly matched ones and
        any still coasting through a brief miss - so the caller can draw a
        stable set of boxes every frame with no on/off flicker.
        """
        centroids = [self._centroid(b) for b in boxes]
        sizes = [self._size(b) for b in boxes]
        assigned_ids = [None] * len(boxes)

        # Candidate matches: close enough in both position AND size
        pairs = []
        for i, (c, s) in enumerate(zip(centroids, sizes)):
            for oid, oc in self.objects.items():
                osz = self.sizes.get(oid, s)
                d = math.hypot(c[0] - oc[0], c[1] - oc[1])
                size_ratio = max(s, osz) / max(1.0, min(s, osz))
                dist_limit = min(self.max_distance, 0.9 * max(s, osz))
                if d <= dist_limit and size_ratio <= self.size_ratio_tolerance:
                    pairs.append((d, i, oid))
        pairs.sort(key=lambda p: p[0])

        matched_boxes = set()
        used_existing_ids = set()
        for d, i, oid in pairs:
            if i in matched_boxes or oid in used_existing_ids:
                continue
            assigned_ids[i] = oid
            matched_boxes.add(i)
            used_existing_ids.add(oid)

        # Update matched tracks with smoothed positions
        a = self.smoothing_alpha
        for i, oid in enumerate(assigned_ids):
            if oid is None:
                continue
            old_box = self.boxes.get(oid, boxes[i])
            smoothed = [a * n + (1 - a) * o for n, o in zip(boxes[i], old_box)]
            self.boxes[oid] = smoothed
            self.objects[oid] = self._centroid(smoothed)
            self.sizes[oid] = self._size(smoothed)
            self.classes[oid] = classes[i]
            self.confs[oid] = confs[i]
            self.missed[oid] = 0

        # Anything unmatched in this frame's detections gets a brand-new ID
        for i in range(len(boxes)):
            if assigned_ids[i] is None:
                new_id = self.next_id
                self.next_id += 1
                self.total_ids_ever += 1
                self.boxes[new_id] = boxes[i]
                self.objects[new_id] = centroids[i]
                self.sizes[new_id] = sizes[i]
                self.classes[new_id] = classes[i]
                self.confs[new_id] = confs[i]
                self.missed[new_id] = 0
                assigned_ids[i] = new_id

        # Age out tracks not matched this frame; drop only after the grace period
        matched_this_frame = set(oid for oid in assigned_ids if oid is not None)
        for oid in list(self.boxes.keys()):
            if oid not in matched_this_frame:
                self.missed[oid] = self.missed.get(oid, 0) + 1
                if self.missed[oid] > self.max_missed_frames:
                    del self.boxes[oid]
                    del self.objects[oid]
                    del self.sizes[oid]
                    del self.classes[oid]
                    del self.confs[oid]
                    del self.missed[oid]

        active = []
        for oid, box in self.boxes.items():
            was_detected = self.missed.get(oid, 0) == 0
            active.append((oid, box, self.classes[oid], self.confs[oid], was_detected))
        return active