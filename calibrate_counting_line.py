"""
Calibration tool: click two points on the first frame of your video to
define a counting line. detection.py can optionally use this line so an
apple is only counted once it crosses it (useful if apples enter/exit
frame repeatedly on a conveyor belt).

USAGE:
    python calibrate_counting_line.py --source videos/input.mp4

Click two points on the preview window (start and end of the line),
then press 's' to save, or 'r' to reset and redraw.
"""

import argparse
import cv2

from utils.config import save_calibration

clicked_points = []


def get_args():
    parser = argparse.ArgumentParser(description="Calibrate the apple counting line")
    parser.add_argument("--source", type=str, default="videos/input.mp4",
                         help="Path to input video to grab a reference frame from")
    return parser.parse_args()


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 2:
        clicked_points.append((x, y))


def main():
    args = get_args()

    cap = cv2.VideoCapture(args.source)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise FileNotFoundError(f"Could not read a frame from '{args.source}'. Check the path.")

    window_name = "Calibrate Counting Line - click 2 points, 's' to save, 'r' to reset, 'q' to quit"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        display = frame.copy()
        for pt in clicked_points:
            cv2.circle(display, pt, 5, (0, 0, 255), -1)
        if len(clicked_points) == 2:
            cv2.line(display, clicked_points[0], clicked_points[1], (0, 0, 255), 2)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            clicked_points.clear()
        elif key == ord("s"):
            if len(clicked_points) == 2:
                save_calibration(clicked_points)
                break
            else:
                print("Click 2 points before saving.")
        elif key == ord("q"):
            print("Calibration cancelled, nothing saved.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()