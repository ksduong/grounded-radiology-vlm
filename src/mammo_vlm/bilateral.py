"""Multi-image extension (notebook 04).

Each training example is the SAME view (CC or MLO) of the LEFT and
RIGHT breast of one patient (same as how radiologists compare).

- view order is always [L, R]
- each box in the target text is tagged with which image/side it belongs to
- each user turn explicitly says the two images are the same patient
- asymmetry is reported per side
"""
from dataclasses import dataclass, field
from pathlib import Path

from .labels import GROUP_TO_KNOWLEDGE, GROUP_TO_PHRASE

_LATERALITY_WORD = {'L': 'left', 'R': 'right'}
_SIDE_ORDER = ['L', 'R']  # consistent order

SYSTEM_MAMMO_MULTIIMAGE = (
    'You are an expert radiology assistant for mammography. '
    'You will be shown the same mammographic view (either CC or MLO) of the LEFT and RIGHT breast of the same patient. '
    'Answer carefully using only the image information provided.'
)


# each instance is an individual image in the dataset
@dataclass(frozen=True)
class MammoView:
    image_id: str
    study_id: str
    laterality: str          # 'L' or 'R'
    view_position: str       # 'CC' or 'MLO'
    png_path: Path
    width: int
    height: int
    birads: str
    density: str
    split: str


# each instance is an annotated finding with its classification/location/side
@dataclass(frozen=True)
class FindingBox:
    group: str          # finding classification
    box: tuple          # (x1, y1, x2, y2), normalized 0-1000
    laterality: str     # 'L' or 'R' (which side this box is from)


@dataclass
class ViewPairGroup:
    """one object = L and R image of the SAME view (CC or MLO) + every finding
    across both, each tagged with which side it's on"""
    study_id: str
    view_position: str
    views: list = field(default_factory=list)     # list of MammoView
    findings: list = field(default_factory=list)  # list of FindingBox

    @property
    def group_id(self):
        return f'{self.study_id}_{self.view_position}'

    @property
    def is_complete_pair(self):
        sides = {v.laterality for v in self.views}
        return {'L', 'R'}.issubset(sides)

    @property
    def split(self):
        return self.views[0].split

    def view_by_laterality(self, laterality):
        for v in self.views:
            if v.laterality == laterality:
                return v
        return None

    # unique finding types present across both sides
    def unique_groups(self):
        return sorted({f.group for f in self.findings})


def build_view_pair_groups(breast_df, image_to_boxes, png_dir, verbose=True):
    """Reshapes per-image data (breast_df rows, image_to_boxes entries) into
    per-(study_id, view_position) L/R pairs. Incomplete pairs are dropped."""
    png_dir = Path(png_dir)
    view_records = {}
    for _, row in breast_df.iterrows():
        view_records[row['image_id']] = MammoView(
            image_id=row['image_id'], study_id=row['study_id'],
            laterality=row['laterality'], view_position=row['view_position'],
            png_path=png_dir / f"{row['image_id']}.png",
            width=int(row['width']), height=int(row['height']),
            birads=str(row['breast_birads']), density=str(row['breast_density']),
            split=row['split'],
        )

    groups = {}
    for v in view_records.values():
        key = (v.study_id, v.view_position)
        if key not in groups:
            groups[key] = ViewPairGroup(study_id=v.study_id, view_position=v.view_position)
        groups[key].views.append(v)

    for g in groups.values():
        g.views.sort(key=lambda v: _SIDE_ORDER.index(v.laterality) if v.laterality in _SIDE_ORDER else 99)

    n_total = len(groups)
    groups = {k: g for k, g in groups.items() if g.is_complete_pair}
    n_dropped = n_total - len(groups)

    # attach bounding box objects to their group
    for image_id, entries in image_to_boxes.items():
        view = view_records.get(image_id)
        if view is None:
            continue
        key = (view.study_id, view.view_position)
        if key not in groups:
            continue
        for e in entries:
            groups[key].findings.append(
                FindingBox(group=e['group'], box=tuple(e['box']), laterality=view.laterality))

    result = list(groups.values())
    if verbose:
        print(f'Built {len(result)} complete L+R view-pair groups (dropped {n_dropped} incomplete)')
    return result


def _ordered_views(group):
    present = {v.laterality: v for v in group.views}
    return [present[side] for side in _SIDE_ORDER if side in present]


def patient_context_sentence(group):
    return (f'Image 1 and Image 2 are the left and right {group.view_position} views, '
            f'respectively, of the same patient.')


def make_multiimage_messages(group, user_text, assistant_text):
    ordered = _ordered_views(group)
    user_content = [{'type': 'image', 'image': str(v.png_path)} for v in ordered]
    user_content.append({'type': 'text', 'text': f'{patient_context_sentence(group)} {user_text}'})
    return [
        {'role': 'system',    'content': [{'type': 'text', 'text': SYSTEM_MAMMO_MULTIIMAGE}]},
        {'role': 'user',      'content': user_content},
        {'role': 'assistant', 'content': [{'type': 'text', 'text': assistant_text}]},
    ]


def _image_index_label(group, laterality):
    ordered = _ordered_views(group)
    v = group.view_by_laterality(laterality)
    if v is None:
        raise ValueError(f'laterality {laterality!r} not found in group {group.group_id}')
    return f'Image {ordered.index(v) + 1} ({_LATERALITY_WORD[laterality]})'


# lists the suspicious findings present; asymmetry gets its side spelled out
def finding_classification_answer(group):
    other_groups = sorted(g for g in group.unique_groups() if g != 'Asymmetry')
    asym_sides = sorted({f.laterality for f in group.findings if f.group == 'Asymmetry'})

    parts = []
    if other_groups:
        parts.append('The suspicious mammographic findings present are: ' + ', '.join(other_groups) + '.')
    if asym_sides:
        sides = ' and '.join(_LATERALITY_WORD[s] for s in asym_sides)
        parts.append(f'An asymmetry is present in the {sides} breast.')
    if not parts:
        return 'No suspicious mammographic finding is present.'
    return ' '.join(parts)


# BI-RADS/density are assigned per breast. L/R pairing -> report both sides explicitly
def birads_density_answer(group):
    parts = []
    for v in _ordered_views(group):
        side = _LATERALITY_WORD[v.laterality]
        parts.append(f'The {side} breast is BI-RADS category {v.birads} with density {v.density}.')
    return ' '.join(parts)


def grounded_report_answer(group):
    parts = [birads_density_answer(group)]
    if not group.findings:
        parts.append('No suspicious mammographic finding is present.')
        return ' '.join(parts)
    for f in group.findings:
        label = _image_index_label(group, f.laterality)
        b = f.box
        if f.group == 'Asymmetry':
            side = _LATERALITY_WORD[f.laterality]
            parts.append(f'An asymmetry is present in the {side} breast, {label}, '
                         f'at <box>{b[0]} {b[1]} {b[2]} {b[3]}</box>.')
        else:
            parts.append(f'{GROUP_TO_PHRASE[f.group]} in {label} <box>{b[0]} {b[1]} {b[2]} {b[3]}</box>.')
    return ' '.join(parts)


def single_finding_grounding_answer(group, finding):
    label = _image_index_label(group, finding.laterality)
    b = finding.box
    if finding.group == 'Asymmetry':
        side = _LATERALITY_WORD[finding.laterality]
        user = f'Compare the two images and locate any asymmetry. {GROUP_TO_KNOWLEDGE["Asymmetry"]}'
        assistant = f'An asymmetry is present in the {side} breast, {label}, at <box>{b[0]} {b[1]} {b[2]} {b[3]}</box>.'
    else:
        user = f'Locate the {finding.group} finding. {GROUP_TO_KNOWLEDGE[finding.group]}'
        assistant = f'The {finding.group} is located in {label} at <box>{b[0]} {b[1]} {b[2]} {b[3]}</box>.'
    return user, assistant


# formats one view-pair group into its list of training examples (same 4 tasks as Model B)
def build_examples_for_group(group):
    if not group.views:
        return []
    base_meta = {
        'group_id': group.group_id, 'study_id': group.study_id, 'view_position': group.view_position,
        'is_complete_pair': group.is_complete_pair,
        'images': [str(v.png_path) for v in _ordered_views(group)],
    }
    examples = []

    # Task 1: finding classification
    examples.append({**base_meta, 'task': 'finding_classification',
        'messages': make_multiimage_messages(group,
            'What suspicious mammographic findings are present, and in which breast?',
            finding_classification_answer(group))})

    # Task 2: BI-RADS + density (per side)
    examples.append({**base_meta, 'task': 'birads_density',
        'messages': make_multiimage_messages(group,
            'What is the BI-RADS category and breast density for each side?',
            birads_density_answer(group))})

    # Task 3: grounded report
    examples.append({**base_meta, 'task': 'grounded_report',
        'messages': make_multiimage_messages(group,
            'Generate a grounded mammography report comparing the left and right images. '
            'For each suspicious finding, state which image and side it appears in and append its '
            'location as <box>x_min y_min x_max y_max</box> in normalized 0-1000 coordinates.',
            grounded_report_answer(group))})

    # Task 4: per-finding grounding
    for finding in group.findings:
        user_instr, assistant_text = single_finding_grounding_answer(group, finding)
        examples.append({**base_meta, 'task': 'single_finding_grounding',
            'messages': make_multiimage_messages(group, user_instr, assistant_text)})

    return examples
