"""Turns free-text model outputs back into labels, and scores them
"""
import re

from .labels import GROUPED_CLASSES

GROUP_KEYWORDS = {
    'Mass': ['mass'],
    'Suspicious Calcification': ['calcification', 'calcifications', 'calcified'],
    'Asymmetry': ['asymmetry'],
    'Architectural Distortion': ['architectural distortion', 'distortion'],
    'Suspicious Lymph Node': ['lymph node'],
    'Other Suspicious Signs': ['other suspicious', 'skin thickening', 'skin retraction', 'nipple retraction'],
}

# if the model explicitly says there's nothing, don't let a stray keyword count as a finding
NEGATIVE_PHRASES = [
    'no suspicious mammographic finding',
    'no suspicious finding',
    'no finding',
    'no suspicious abnormality',
    'no mammographic evidence of malignancy',
]

# NOTE on the \b at the end of both patterns:
# VinDr stores labels as 'BI-RADS 3' / 'DENSITY B', so the bilateral targets read
# "...BI-RADS category BI-RADS 3 with density DENSITY B." Without \b, the density
# regex matches the first "density D" (the D of "DENSITY") and every case is
# parsed as density D. The single-image targets ("The breast density is DENSITY B.")
# never hit this, which is why it only showed up in the multi-image eval.
BIRADS_RE = re.compile(r'bi-?rads\s*(?:category\s*)?([1-5])\b', re.I)
DENSITY_RE = re.compile(r'density\s*([abcd])\b', re.I)
BOX_RE = re.compile(r'<box>\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*</box>', re.I)


def extract_findings(text, negation_aware=True):
    t = str(text).lower()
    if negation_aware and any(p in t for p in NEGATIVE_PHRASES):
        return set()
    out = set()
    for g, kws in GROUP_KEYWORDS.items():
        if any(k in t for k in kws):
            out.add(g)
    return out


def to_binary_vec(groups):
    return [1 if g in groups else 0 for g in GROUPED_CLASSES]


def extract_birads(text):
    m = BIRADS_RE.search(str(text))
    return m.group(1) if m else None


def extract_density(text):
    m = DENSITY_RE.search(str(text))
    return m.group(1).upper() if m else None


def extract_boxes(text):
    return [list(map(int, m)) for m in BOX_RE.findall(str(text))]


def _sentence_for_side(text, side_word):
    # split after a period that ends a sentence; 'BI-RADS 3.' still ends cleanly
    for s in re.split(r'(?<=\.)\s+', str(text)):
        if f'{side_word} breast' in s.lower():
            return s
    return None


def extract_birads_density_by_side(text):
    """For bilateral outputs: {'L': (birads, density), 'R': (birads, density)}.
    A side the model never mentions comes back as (None, None)."""
    out = {}
    for side, word in [('L', 'left'), ('R', 'right')]:
        s = _sentence_for_side(text, word)
        out[side] = (extract_birads(s), extract_density(s)) if s else (None, None)
    return out


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0
