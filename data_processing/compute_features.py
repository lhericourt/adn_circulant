from pathlib import Path
from typing import List
import pickle
import argparse

import pandas as pd
from sklearn.model_selection import StratifiedKFold


MERGED_DIR = Path('.') / 'data' / 'merged'
TRAINING_DIR = Path('.') / 'data' / 'for_training'
OUTLIERS_ID = ['OS2006_34', 'OS2006_548', 'OS2006_20']  # CPA numbers too high
MIN_DAYS_DURATION_OBSERVATION = 700


def get_patient_with_small_timeos(data: pd.DataFrame, min_days: int) -> List[str]:
    return data.loc[(data['time_OS'] < min_days) & (data['relapse'] == 0), 'patient_id'].tolist()


def filter_data(data: pd.DataFrame, outliers_id: List[str]) -> pd.DataFrame:
    # it is important to keep the indexes are they are and not to reset them, to be able to merge easily features with
    # metadata afterwards
    return data[~data['patient_id'].isin(outliers_id)]


def remove_x_and_y_chrom(features: pd.DataFrame) -> pd.DataFrame:
    return features[[x for x in features.columns if not (x.startswith('X:') or x.startswith('Y:'))]]


def compute_features(data: pd.DataFrame, score: str = 'ratio') -> pd.DataFrame:
    data.set_index('patient_id', inplace=True)
    features = data[[x for x in data.columns if f'_{score}' in x]]
    features = remove_x_and_y_chrom(features)
    features.fillna(0, inplace=True)
    return features


def save_features_and_labesl(X_train: pd.DataFrame, X_test: pd.DataFrame,
                            y_train: pd.Series, y_test: pd.Series,
                            cv_nb: int, folder: Path) -> None:
    pickle.dump(X_train, open(folder / f'X_train_{cv_nb}.pkl', 'wb'))
    pickle.dump(X_test, open(folder / f'X_test_{cv_nb}.pkl', 'wb'))
    pickle.dump(y_train, open(folder / f'y_train_{cv_nb}.pkl', 'wb'))
    pickle.dump(y_test, open(folder / f'y_test_{cv_nb}.pkl', 'wb'))


def split_data_in_K_folds(features: pd.DataFrame, labels: pd.Series, nb_splits: int) -> None:
    random_state = 42
    skf = StratifiedKFold(nb_splits, shuffle=True, random_state=random_state)

    for i, (train_index, test_index) in enumerate(skf.split(features, labels)):
        X_train, X_test = features.iloc[train_index, :], features.iloc[test_index, :]
        y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
        save_features_and_labesl(X_train, X_test, y_train, y_test, cv_nb=i, folder=TRAINING_DIR)


def compute_features_for_all_folds(dataset_filename: str, score: str = 'ratio', label_column: str = 'relapse'):
    nb_splits = 5

    data = pd.read_csv(MERGED_DIR / dataset_filename)
    patient_ids_to_remove = get_patient_with_small_timeos(data, MIN_DAYS_DURATION_OBSERVATION)
    patient_ids_to_remove.extend(OUTLIERS_ID)
    data = filter_data(data, patient_ids_to_remove)

    labels = data.set_index('patient_id')[label_column]
    features = compute_features(data, score)

    split_data_in_K_folds(features, labels, nb_splits)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-df', '--dataset_filename',
                        help='Dataset file to use, chir_data.csv, diag_data.csv or end_data.csv',
                        default='diag_data.csv')
    parser.add_argument('-s', '--score',
                        help='score to use in features, ratio or zscore',
                        default='ratio')
    parser.add_argument('-l', '--label',
                        help='column name to use as label, relapse or etat2',
                        default='relapse')

    args = parser.parse_args()

    compute_features_for_all_folds(args.dataset_filename, args.score, args.label)


