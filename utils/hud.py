"""
Visual layer: color-by-class boxes (every apple of the same class shares
the same color; different classes get visually distinct colors), plus a
right-hand HUD panel with a per-class + total count breakdown and a
sparkline graph of the running total.
"""

import colorsys
import cv2
import numpy as np

# Fixed, eye-catching palette - assigned in class-id order so the same
# class always gets the same color across runs.
_PALETTE = [
    (60, 255, 60),    # bright green - class 0 (apple) - pops clearly against red apples
    (50, 50, 255),    # red - class 1 (damaged_apple) - green=good, red=damaged
    (255, 255, 0),    # cyan
    (40, 200, 235),   # yellow
    (235, 60, 60),    # electric blue
    (180, 20, 255),   # hot pink
]


def build_class_colors(names_dict):
    """names_dict: model.names, e.g. {0: 'apple', 1: 'damaged_apple'}.
    Returns {class_name: (B, G, R)}, stable across runs."""
    colors = {}
    for i, cid in enumerate(sorted(names_dict.keys())):
        colors[names_dict[cid]] = _PALETTE[i % len(_PALETTE)]
    return colors


def draw_apple_box(frame, x1, y1, x2, y2, track_id, conf, class_name, color, show_conf=False):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, lineType=cv2.LINE_AA)

    label = f"#{track_id} {class_name} {conf:.2f}" if show_conf else f"#{track_id}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(frame, (x1, y1 - th - 5), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def draw_hud_panel(frame, total_count, current_count, history, class_counts,
                    class_colors, panel_width=250, avg_conf=0.0, fps=0.0):
    """
    class_counts: {class_name: count} - unique count per class so far
    class_colors: {class_name: (B, G, R)} - matches box colors, used as
                  a legend so the panel doubles as a color key
    """
    h, w = frame.shape[:2]
    x0 = max(0, w - panel_width)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, 0), (w, h), (20, 20, 20), -1)
    frame[:] = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    pad = 16
    y = pad + 10

    cv2.putText(frame, "APPLE COUNTER", (x0 + pad, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 255, 60), 2, cv2.LINE_AA)
    y += 12
    cv2.line(frame, (x0 + pad, y), (w - pad, y), (80, 80, 80), 1)
    y += 45

    cv2.putText(frame, str(total_count), (x0 + pad, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (60, 255, 60), 3, cv2.LINE_AA)
    y += 15
    cv2.putText(frame, "TOTAL COUNTED", (x0 + pad, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    y += 30

    # Per-class breakdown (color swatch + name + count) - acts as legend
    cv2.line(frame, (x0 + pad, y), (w - pad, y), (60, 60, 60), 1)
    y += 20
    for class_name, count in class_counts.items():
        color = class_colors.get(class_name, (255, 255, 255))
        cv2.rectangle(frame, (x0 + pad, y - 11), (x0 + pad + 14, y + 1), color, -1)
        cv2.putText(frame, f"{class_name}: {count}", (x0 + pad + 22, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
        y += 22

    y += 8
    cv2.line(frame, (x0 + pad, y), (w - pad, y), (60, 60, 60), 1)
    y += 22

    cv2.putText(frame, f"In frame now: {current_count}", (x0 + pad, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    y += 22
    cv2.putText(frame, f"Avg confidence: {avg_conf:.0%}", (x0 + pad, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    y += 22
    cv2.putText(frame, f"Speed: {fps:.1f} fps", (x0 + pad, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    y += 28

    cv2.putText(frame, "COUNT TREND", (x0 + pad, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    y += 10
    graph_h = 65
    graph_top = y + 8
    graph_bottom = graph_top + graph_h
    graph_left = x0 + pad
    graph_right = w - pad

    cv2.rectangle(frame, (graph_left, graph_top), (graph_right, graph_bottom), (60, 60, 60), 1)

    if len(history) >= 2:
        vals = np.array(history[-80:], dtype=np.float32)
        v_min, v_max = float(vals.min()), float(vals.max())
        span = max(1.0, v_max - v_min)
        n = len(vals)
        gw = graph_right - graph_left
        pts = []
        for i, v in enumerate(vals):
            px = graph_left + int(i / max(1, n - 1) * gw)
            py = graph_bottom - int((v - v_min) / span * graph_h)
            pts.append((px, py))
        for i in range(len(pts) - 1):
            cv2.line(frame, pts[i], pts[i + 1], (60, 255, 60), 2, cv2.LINE_AA)

    return frame