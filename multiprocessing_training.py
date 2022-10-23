from pathlib import Path
import pickle
import logging
from logging.handlers import RotatingFileHandler
import argparse

from random import sample
from multiprocessing import Pool
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-fn', '--folds_number',
                    help='Fold number to use, from 0 to 4',
                    default='0')
parser.add_argument('-p', '--folder_path',
                    help='Folder where features and labels are stored',
                    default='data/for_training')
parser.add_argument('-pn', '--processors_number',
                    help='Number of processors to use',
                    default=2)

args = parser.parse_args()

app_log = logging.getLogger('root')
formatter = logging.Formatter('%(message)s')
my_handler = RotatingFileHandler(f'adn_circulant_brute_force_fold_{args.folds_number}.log',
                                 mode='a', maxBytes=100*1024*1024)
my_handler.setFormatter(formatter)
my_handler.setLevel(logging.INFO)
app_log.setLevel(logging.INFO)
app_log.addHandler(my_handler)

X = pickle.load(open(Path(args.folder_path) / f'X_train_{args.folds_number}.pkl', 'rb'))
y = pickle.load(open(Path(args.folder_path) / f'y_train_{args.folds_number}.pkl', 'rb'))

features_name = X.columns.tolist()
nb_features_to_pick = 200


def compute_model_with_random_features(i):
    features_to_use = sample(features_name, nb_features_to_pick)
    lr = LogisticRegression()
    cv_score = cross_val_score(lr, X[features_to_use], y, cv=5)
    mean_score = round(100 * np.mean(cv_score), 1)
    std_score = round(100 * np.std(cv_score), 1)
    app_log.warning(f'{mean_score}|{std_score}|{cv_score}|{features_to_use}')


if __name__ == '__main__':

    with Pool(int(args.processors_number)) as p:
        list(tqdm(p.imap(compute_model_with_random_features, range(3_000_000), chunksize=10), total=100))
