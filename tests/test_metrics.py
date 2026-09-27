import re

from mammo_vlm.metrics import (extract_birads, extract_birads_density_by_side, extract_density,
                               extract_findings, iou)

SINGLE_GT = 'The BI-RADS category is BI-RADS 3. The breast density is DENSITY B.'
BILATERAL_GT = ('The left breast is BI-RADS category BI-RADS 1 with density DENSITY B. '
                'The right breast is BI-RADS category BI-RADS 4 with density DENSITY C.')


def test_single_image_parsing():
    assert extract_birads(SINGLE_GT) == '3'
    assert extract_density(SINGLE_GT) == 'B'


def test_old_density_regex_breaks_on_bilateral_targets():
    # regression test for the bug: the original pattern (no \b) reads "density D(ENSITY)"
    old = re.compile(r'density\s*([abcd])', re.I)
    assert old.search(BILATERAL_GT).group(1).upper() == 'D'
    assert extract_density(BILATERAL_GT) == 'B'


def test_per_side_extraction():
    assert extract_birads_density_by_side(BILATERAL_GT) == {'L': ('1', 'B'), 'R': ('4', 'C')}


def test_missing_side_is_none():
    out = extract_birads_density_by_side('The left breast is BI-RADS category BI-RADS 2 with density DENSITY A.')
    assert out['R'] == (None, None)


def test_negation_overrides_keywords():
    assert extract_findings('No suspicious mammographic finding is present.') == set()
    assert extract_findings('A suspicious mass is present <box>1 2 3 4</box>.') == {'Mass'}


def test_iou():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert abs(iou([0, 0, 10, 10], [5, 0, 15, 10]) - 1 / 3) < 1e-9
