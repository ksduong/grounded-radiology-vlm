"""Bounding-box scaling

VinDr boxes in original DICOM pixel coords. The PNGs we train on can be a
different size, so: DICOM coords -> PNG pixel coords -> normalized 0-1000
(the integer grid Qwen2.5-VL uses inside <box>...</box> tokens).
"""


def box_to_norm1000(row, png_size, min_px=2):
    """row needs xmin/ymin/xmax/ymax + width/height (original DICOM size).
    png_size is (W, H) of the PNG.
    Returns [x1, y1, x2, y2] ints in 0-1000, or None if the box is degenerate."""
    if png_size is None:
        return None
    png_w, png_h = png_size

    orig_w, orig_h = float(row['width']), float(row['height'])
    sx, sy = png_w / orig_w, png_h / orig_h

    x1 = float(row['xmin']) * sx
    y1 = float(row['ymin']) * sy
    x2 = float(row['xmax']) * sx
    y2 = float(row['ymax']) * sy

    # clip to the image, drop boxes that collapse to (almost) nothing
    x1 = max(0, min(png_w, x1)); x2 = max(0, min(png_w, x2))
    y1 = max(0, min(png_h, y1)); y2 = max(0, min(png_h, y2))
    if x2 - x1 < min_px or y2 - y1 < min_px:
        return None

    return [
        int(round(x1 / png_w * 1000)),
        int(round(y1 / png_h * 1000)),
        int(round(x2 / png_w * 1000)),
        int(round(y2 / png_h * 1000)),
    ]


def denorm_box(box, W, H):
    """box is [x1, y1, x2, y2] normalized 0-1000 -> pixel coords for this image."""
    x1, y1, x2, y2 = box
    return (x1 / 1000 * W, y1 / 1000 * H, x2 / 1000 * W, y2 / 1000 * H)
