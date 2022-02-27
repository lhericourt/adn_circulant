from pathlib import Path
import pickle
import logging
from logging.handlers import RotatingFileHandler

from random import sample
from multiprocessing import Pool
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np

app_log = logging.getLogger('root')
formatter = logging.Formatter('%(message)s')
my_handler = RotatingFileHandler('adn_circulant_brute_force.log', mode='a', maxBytes=100*1024*1024)
my_handler.setFormatter(formatter)
my_handler.setLevel(logging.INFO)
app_log.setLevel(logging.INFO)
app_log.addHandler(my_handler)

X = pickle.load(open(Path('features') / 'X.pkl', 'rb'))
y = pickle.load(open(Path('features') / 'y.pkl', 'rb'))

features_name = X.columns.tolist()
nb_features_to_pick = 200


def compute_model_with_random_features(i):
    features_to_use = sample(features_name, nb_features_to_pick)
    lr = LogisticRegression()
    cv_score = cross_val_score(lr, X[features_to_use], y, cv=5)
    mean_score = round(100 * np.mean(cv_score), 1)
    std_score = round(100 * np.std(cv_score), 1)
    app_log.warning(f'{mean_score}|{std_score}|{cv_score}')


if __name__ == '__main__':
    with Pool() as p:
        list(tqdm(p.imap(compute_model_with_random_features, range(100), chunksize=10), total=100))