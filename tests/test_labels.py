import pandas as pd

from mammo_vlm.labels import build_image_to_groups, map_to_groups, parse_finding_categories


def test_parses_stringified_list():
    assert parse_finding_categories("['Mass', 'Focal Asymmetry']") == ['Mass', 'Focal Asymmetry']


def test_nan_is_empty():
    assert parse_finding_categories(float('nan')) == []


def test_asymmetry_subtypes_collapse():
    assert map_to_groups("['Focal Asymmetry', 'Global Asymmetry']") == ['Asymmetry']


def test_no_finding_wins():
    assert map_to_groups("['No Finding']") == []


def test_ungrouped_category_ignored():
    # categories outside GROUPS (e.g. benign-looking ones) shouldn't create a label
    assert map_to_groups("['Something Else']") == []


def test_image_to_groups_merges_rows():
    df = pd.DataFrame({
        'image_id': ['a', 'a', 'b'],
        'finding_categories': ["['Mass']", "['Skin Thickening']", "['No Finding']"],
    })
    assert build_image_to_groups(df) == {'a': ['Mass', 'Other Suspicious Signs']}
