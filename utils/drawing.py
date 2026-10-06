"""
Drawing helpers: bounding boxes, labels, and the running total-count overlay.
Kept separate from detection.py so detection logic stays clean and each
piece is easy to test/reuse on its own.
"""

import cv2


def draw_box(frame, x1, y1, x2, y2, track_id, conf, label_name="Apple",
             box_color=(255, 0, 0), text_color=(255, 255, 255)):
    """Draw a single detection box with an 'id:xxx Label conf' tag."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    label = f"id:{track_id} {label_name} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), box_color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    return frame


def draw_total_count(frame, count, height, position="bottom-left",
                      bg_color=(0, 255, 255), text_color=(0, 0, 0)):
    """Draw the 'Total Count: N' banner, matching the reference demo style."""
    count_text = f"Total Count: {count}"
    (tw, th), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)

    if position == "bottom-left":
        x, y = 10, height - 10
    else:  # top-left fallback
        x, y = 10, th + 20

    cv2.rectangle(frame, (x, y - th - 20), (x + tw + 10, y), bg_color, -1)
    cv2.putText(frame, count_text, (x + 5, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2, cv2.LINE_AA)

    return frame


def draw_counting_line(frame, line_points, color=(80, 255, 150), thickness=3):
    """Draws the counting line prominently, since apples are only counted
    when they cross it (not just when detected)."""
    if line_points and len(line_points) == 2:
        p1, p2 = tuple(line_points[0]), tuple(line_points[1])
        cv2.line(frame, p1, p2, (0, 0, 0), thickness + 4, cv2.LINE_AA)  # dark outline for contrast
        cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)
    return frame