"""Study-level train/val/test splits.
"""
import random


def make_study_splits(breast_df, seed=0, val_frac=0.10):
    train_studies_all = sorted(breast_df.loc[breast_df['split'] == 'training', 'study_id'].unique().tolist())
    test_studies = sorted(breast_df.loc[breast_df['split'] == 'test', 'study_id'].unique().tolist())

    # fixed seed -> same shuffle every run, so every model sees the same val/test studies
    rng = random.Random(seed)
    rng.shuffle(train_studies_all)
    n_val = int(len(train_studies_all) * val_frac)
    val_studies = sorted(train_studies_all[:n_val])
    train_studies = sorted(train_studies_all[n_val:])

    # sanity: no study in two splits
    assert not (set(train_studies) & set(val_studies))
    assert not (set(train_studies) & set(test_studies))
    assert not (set(val_studies) & set(test_studies))

    def studies_to_images(study_ids):
        return sorted(breast_df.loc[breast_df['study_id'].isin(study_ids), 'image_id'].unique().tolist())

    return {
        'train_studies': train_studies, 'val_studies': val_studies, 'test_studies': test_studies,
        'train_ids': studies_to_images(train_studies),
        'val_ids': studies_to_images(val_studies),
        'test_ids': studies_to_images(test_studies),
    }
