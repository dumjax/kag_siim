import pandas as pd
from sklearn import model_selection
import yaml


# create folds
def folds_generator(nr_folds):
    df = pd.read_csv("../data/raw/train.csv")
    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.target.values
    kf = model_selection.StratifiedKFold(n_splits=nr_folds)
    
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f
    'train_folds_{}.csv'.format(NR_FOLDS)
    df.to_csv('train_folds_{}.csv'.format(NR_FOLDS), index=False)


def load_yaml(file_name):
    with open(file_name, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.SafeLoader)

    return config
