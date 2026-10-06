"""
Line-crossing detection.

Instead of counting an apple the moment it's first detected (which counts
everything visible, even apples that never actually pass a checkpoint),
this checks which side of a reference line each tracked apple is on and
only counts it the moment it crosses from one side to the other - the
same idea as a real conveyor-belt counting sensor.
"""


def side_of_line(point, line_p1, line_p2):
    """Returns +1, -1, or 0 depending on which side of the line the point is on."""
    px, py = point
    x1, y1 = line_p1
    x2, y2 = line_p2
    cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    if cross > 0:
        return 1
    if cross < 0:
        return -1
    return 0


def default_line(width, height, y_fraction=0.55, tilt=0.25):
    """A full-width line used when no calibration file exists.
    tilt: fraction of frame height difference between the left and right
    ends (0 = perfectly horizontal, higher = steeper diagonal), matching
    the angled look of a real conveyor-belt counting line."""
    y_center = height * y_fraction
    y_offset = height * tilt / 2
    left_y = int(y_center - y_offset)
    right_y = int(y_center + y_offset)
    return [(0, left_y), (width, right_y)]