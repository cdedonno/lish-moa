import pandas as pd
from sklearn.model_selection import KFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

SEED = 42
FOLDS = 5

def create_folds(n_splits, seed):

    train_features = pd.read_csv('./data/train_features.csv')
    train_targets_scored = pd.read_csv('./data/train_targets_scored.csv')
    train_drug = pd.read_csv('./data/train_drug.csv')

    # Get rid of "ctl_vehicle" from training.
    # You may comment below lines if you do not want to do it.
    train_targets_scored = train_targets_scored.loc[train_features['cp_type'] == 'trt_cp', :]
    train_features = train_features[train_features['cp_type'] == 'trt_cp']
    train_features_drug = train_features.merge(train_drug, on="sig_id", how='left')

    # Add drug_id as one of the targets (for stratifying later)
    targets = train_targets_scored.columns[1:]
    train_targets_scored = train_targets_scored.merge(train_drug, on='sig_id', how='left')

    # Within training data, identify indices where drug ids
    # which are present in more than 18 rows and less than 18 rows
    vc = train_targets_scored.drug_id.value_counts()
    vc1 = vc.loc[vc <= 18].index
    vc2 = vc.loc[vc > 18].index

    # tmp is a dataframe derived from scored targets, where targets are
    # averaged by drugid (one row per drug id)
    tmp = train_targets_scored.groupby('drug_id')[targets].mean().loc[vc1]
    tmp = tmp.reset_index()
    tmp = tmp.rename(columns={"index":"drug_id"})

    # tmp1 is a dataframe with targets and drug_id for all drugs that have
    # repeated more that 18 times in train dataset.
    # We are stratifying these drugs as among all folds.
    # Thought here is that such drugs might repeat in public/private test sets as well
    tmp1 = train_targets_scored[train_targets_scored['drug_id'].isin(vc2)]
    tmp1 = tmp1.reset_index(drop=True)

    skf = MultilabelStratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed
    )
    tmp_copy = tmp.copy()
    tmp1_copy = tmp1.copy()
    train_indices = train_features_drug[['sig_id', 'drug_id']].copy()

    for fold, (_, idxV) in enumerate(skf.split(X=tmp_copy, y=tmp_copy[targets])):
        tmp_copy.loc[idxV, "kfold"] = fold
    train_indices = train_indices.merge(
        tmp_copy[['drug_id', 'kfold']],
        on='drug_id',
        how="left"
    )

    for fold,(_, idxV) in enumerate(skf.split(X=tmp1_copy, y=tmp1_copy[targets])):
        tmp1_copy.loc[idxV, "kfold"] = fold      
    train_indices = train_indices.merge(
        tmp1_copy[['sig_id', 'kfold']],
        on='sig_id',
        how="left"
    )

    train_indices['kfold'] = train_indices['kfold_x'].combine_first(train_indices['kfold_y'])  
    train_indices.drop(['drug_id', 'kfold_x', 'kfold_y'], inplace=True, axis=1)
    return train_indices
