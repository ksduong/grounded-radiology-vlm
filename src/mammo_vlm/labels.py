"""Finding-label grouping for VinDr-Mammo.
"""
import ast
from collections import defaultdict

import pandas as pd

GROUPS = {
    'Mass':                     ['Mass'],
    'Suspicious Calcification': ['Suspicious Calcification'],
    'Asymmetry':                ['Asymmetry', 'Focal Asymmetry', 'Global Asymmetry'],
    'Architectural Distortion': ['Architectural Distortion'],
    'Suspicious Lymph Node':    ['Suspicious Lymph Node'],
    'Other Suspicious Signs':   ['Skin Thickening', 'Skin Retraction', 'Nipple Retraction'],
}
GROUPED_CLASSES = list(GROUPS.keys())

GROUP_TO_PHRASE = {
    'Mass': 'A suspicious mass is present',
    'Suspicious Calcification': 'Suspicious calcifications are present',
    'Asymmetry': 'An asymmetry is present',
    'Architectural Distortion': 'Architectural distortion is present',
    'Suspicious Lymph Node': 'A suspicious lymph node is present',
    'Other Suspicious Signs': 'Other suspicious mammographic signs are present',
}

# "knowledge" sentence appended to the per-finding grounding prompt
GROUP_TO_KNOWLEDGE = {
    'Mass': 'A mammographic mass is a space-occupying lesion with a visible shape, margin, and density.',
    'Suspicious Calcification': 'Suspicious calcifications appear as small high-density bright foci, often grouped or clustered.',
    'Asymmetry': 'Asymmetry is an area of denser breast tissue without a clear mass, possibly focal or global.',
    'Architectural Distortion': 'Architectural distortion appears as pulling or disruption of normal breast tissue without a definite mass.',
    'Suspicious Lymph Node': 'A suspicious lymph node appears enlarged, dense, rounded, or with abnormal morphology.',
    'Other Suspicious Signs': 'Other suspicious signs include skin thickening, nipple retraction, or skin retraction.',
}


# fix inconsistencies in the finding_categories col of finding_df
# (it's stored as a stringified python list, e.g. "['Mass', 'Suspicious Calcification']")
def parse_finding_categories(x):
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    try:
        v = ast.literal_eval(str(x))
        return v if isinstance(v, list) else [str(v)]
    except Exception:
        return [str(x)]


# takes clean finding list and maps it to the classes defined in GROUPS,
# returning a sorted list of unique groups
def map_to_groups(categories):
    cats = parse_finding_categories(categories)
    if 'No Finding' in cats:
        return []
    out = set()
    for g, raws in GROUPS.items():
        for r in raws:
            if r in cats:
                out.add(g)
    return sorted(out)


def build_image_to_groups(finding_df):
    """image_id -> sorted list of grouped labels. Tells WHAT is in each image."""
    image_to_groups = defaultdict(set)
    for _, r in finding_df.iterrows():
        for g in map_to_groups(r['finding_categories']):
            image_to_groups[r['image_id']].add(g)
    return {k: sorted(v) for k, v in image_to_groups.items()}
