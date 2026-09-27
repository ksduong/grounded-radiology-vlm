import pandas as pd

from mammo_vlm.splits import make_study_splits


def _fake_breast_df(n_train=50, n_test=10):
    rows = []
    for i in range(n_train + n_test):
        split = 'training' if i < n_train else 'test'
        for lat in 'LR':
            for view in ('CC', 'MLO'):
                rows.append({'study_id': f's{i}', 'image_id': f's{i}_{lat}{view}', 'split': split})
    return pd.DataFrame(rows)


def test_all_views_of_a_study_stay_together():
    df = _fake_breast_df()
    sp = make_study_splits(df)
    for name in ('train', 'val', 'test'):
        studies = {i.split('_')[0] for i in sp[f'{name}_ids']}
        assert studies == set(sp[f'{name}_studies'])
        assert len(sp[f'{name}_ids']) == 4 * len(studies)


def test_val_size_and_official_test_untouched():
    sp = make_study_splits(_fake_breast_df(), val_frac=0.10)
    assert len(sp['val_studies']) == 5
    assert sp['test_studies'] == sorted(f's{i}' for i in range(50, 60))


def test_same_seed_same_split():
    df = _fake_breast_df()
    assert make_study_splits(df, seed=0) == make_study_splits(df, seed=0)
    assert make_study_splits(df, seed=0)['val_studies'] != make_study_splits(df, seed=1)['val_studies']
