import pandas as pd
import pytest

from mammo_vlm.bilateral import (birads_density_answer, build_examples_for_group,
                                 build_view_pair_groups, finding_classification_answer,
                                 grounded_report_answer)


def _breast_df(drop=None):
    rows = []
    for lat, birads, dens in [('R', 'BI-RADS 4', 'DENSITY C'), ('L', 'BI-RADS 1', 'DENSITY C')]:
        for view in ('CC', 'MLO'):
            rows.append({'image_id': f'{lat}{view}', 'study_id': 's1', 'laterality': lat,
                         'view_position': view, 'width': 3000, 'height': 4000,
                         'breast_birads': birads, 'breast_density': dens, 'split': 'training'})
    df = pd.DataFrame(rows)
    return df[df['image_id'] != drop] if drop else df


BOXES = {
    'RCC': [{'group': 'Mass', 'box': [10, 20, 30, 40]}],
    'LCC': [{'group': 'Asymmetry', 'box': [50, 60, 70, 80]}],
}


def _cc_group(**kw):
    groups = build_view_pair_groups(_breast_df(**kw), BOXES, png_dir='png', verbose=False)
    return {g.view_position: g for g in groups}


def test_one_group_per_view_and_left_is_always_image_1():
    groups = _cc_group()
    assert set(groups) == {'CC', 'MLO'}
    assert [v.laterality for v in groups['CC'].views] == ['L', 'R']  # R was listed first in the df


def test_incomplete_pair_is_dropped():
    assert set(_cc_group(drop='LMLO')) == {'CC'}


def test_boxes_tagged_with_correct_image_and_side():
    report = grounded_report_answer(_cc_group()['CC'])
    assert 'A suspicious mass is present in Image 2 (right) <box>10 20 30 40</box>.' in report
    assert 'asymmetry is present in the left breast, Image 1 (left), at <box>50 60 70 80</box>' in report


def test_mlo_group_has_no_findings():
    g = _cc_group()['MLO']
    assert finding_classification_answer(g) == 'No suspicious mammographic finding is present.'


def test_birads_reported_per_side():
    ans = birads_density_answer(_cc_group()['CC'])
    assert ans.startswith('The left breast is BI-RADS category BI-RADS 1')
    assert 'The right breast is BI-RADS category BI-RADS 4' in ans


def test_example_count_is_3_plus_one_per_finding():
    exs = build_examples_for_group(_cc_group()['CC'])
    assert [e['task'] for e in exs].count('single_finding_grounding') == 2
    assert len(exs) == 5
    user = exs[0]['messages'][1]['content']
    assert [c['type'] for c in user] == ['image', 'image', 'text']
    assert 'same patient' in user[-1]['text']
