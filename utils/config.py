"""
Small helper for saving/loading the calibration data (counting line, ROI)
produced by calibrate_counting_line.py, so detection.py can reuse it.
"""

import json
import os

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "calibration.json")


def save_calibration(line_points, path=DEFAULT_CONFIG_PATH):
    data = {"counting_line": line_points}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Calibration saved to {path}")


def load_calibration(path=DEFAULT_CONFIG_PATH):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("counting_line")