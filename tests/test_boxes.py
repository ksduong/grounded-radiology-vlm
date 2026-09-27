from mammo_vlm.boxes import box_to_norm1000, denorm_box


def _row(xmin, ymin, xmax, ymax, w=3000, h=4000):
    return {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax, 'width': w, 'height': h}


def test_same_size_png_is_just_normalization():
    assert box_to_norm1000(_row(300, 400, 600, 800), png_size=(3000, 4000)) == [100, 100, 200, 200]


def test_downsized_png_gives_same_normalized_box():
    # the whole reason for the PNG scaling step: resizing the image must not move the box
    full = box_to_norm1000(_row(300, 400, 600, 800), png_size=(3000, 4000))
    half = box_to_norm1000(_row(300, 400, 600, 800), png_size=(1500, 2000))
    assert full == half


def test_box_outside_image_gets_clipped():
    assert box_to_norm1000(_row(-50, -50, 600, 800), png_size=(3000, 4000)) == [0, 0, 200, 200]


def test_degenerate_box_dropped():
    assert box_to_norm1000(_row(100, 100, 101, 500), png_size=(3000, 4000)) is None


def test_missing_png_returns_none():
    assert box_to_norm1000(_row(1, 1, 50, 50), png_size=None) is None


def test_denorm_roundtrip():
    assert denorm_box([100, 100, 200, 200], 1500, 2000) == (150, 200, 300, 400)
