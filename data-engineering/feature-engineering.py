import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline as make_sequence
from sklearn.pipeline import make_union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
import re, os
from joblib import Parallel, delayed
from sklearn.pipeline import FeatureUnion, _fit_transform_one, _transform_one, _name_estimators
from scipy import sparse
import argparse
import yaml
from dkube.sdk import *

import random
random.seed(42)


inpath = "/data/clean"
test_fs_path = "/data/test_fs"
train_fs_path = "/data/train_fs"

# it simply takes the name of the column we want to extract and if we use it, 
# it will 'spit out' the data column of our Data Frame.
class FeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(self, feature_names):
        self.feature_names = feature_names
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        return X[self.feature_names]


def extract_items(list_, key, all_=True):
    def sub(x):
        return re.sub(r'[^A-Za-z0-9]', '_', x)
    if all_:
        target = []
        for dict_ in eval(list_):
            target.append(sub(dict_[key].strip()))
        return ' '.join(target)
    elif not eval(list_):
        return 'no_data'
    else:
        return sub(eval(list_)[0][key].strip())

# This one is a bit more complex. It's role is to:
# 1st - extract values from dictionaries,
# 2nd - join them in one string,
# 3rd - dummify it using sklearn Count Vectorizer.
class DictionaryVectorizer(BaseEstimator, TransformerMixin):
    
    def __init__(self, key, all_=True):
        self.key = key
        self.all = all_
    
    def fit(self, X, y=None):
        def lam_genres(x):
            return extract_items(x, self.key, self.all)
        genres = X.apply(lam_genres)
        self.vectorizer = CountVectorizer().fit(genres)        
        self.columns = self.vectorizer.get_feature_names()
        return self
        
    def transform(self, X):
        def lam2_generes(x):
            return extract_items(x, self.key)
        genres = X.apply(lam2_generes)
        data = self.vectorizer.transform(genres)
        return pd.DataFrame(data.toarray(), columns=self.vectorizer.get_feature_names(), index=X.index)

# This transformer expects dummified data set and extract most popular features.
class TopFeatures(BaseEstimator, TransformerMixin):
    
    def __init__(self, percent):
        if percent > 100:
            self.percent = 100
        else:
            self.percent = percent
    
    def fit(self, X, y=None):
        counts = X.sum().sort_values(ascending=False)
        index_ = int(counts.shape[0]*self.percent/100)
        self.columns = counts[:index_].index
        return self
    
    def transform(self, X):
        return X[self.columns]

# Sum Transformer simply computes a sum across given features.
class SumTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, series_name):
        self.series_name = series_name
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        return X.sum(axis=1).to_frame(self.series_name)

# Biniarizer takes as an input function that decides whether or not label value as True or False.
class Binarizer(BaseEstimator, TransformerMixin):
    
    def __init__(self, condition, name):
        self.condition = condition
        self.name = name
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        def lam_binarizer(x):
            return int(self.condition(x))
        return X.apply(lam_binarizer).to_frame(self.name)


from datetime import datetime

def get_year(date):
    return datetime.strptime(date, '%Y-%m-%d').year

def get_month(date):
    return datetime.strptime(date, '%Y-%m-%d').strftime('%b')

def get_weekday(date):
    return datetime.strptime(date, '%Y-%m-%d').strftime('%a')

# As mentioned earlier, this transformer takes a date in string format and extract values of interest.
class DateTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        year = X.apply(get_year).rename('year')
        month = pd.get_dummies(X.apply(get_month))
        day = pd.get_dummies(X.apply(get_weekday))
        return pd.concat([year, month, day], axis=1)        


def get_list_len(list_):
    return len(eval(list_))

# Item Counter counts how many items are in a list.
class ItemCounter(BaseEstimator, TransformerMixin):
        
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        def lam_itemcounter(x):
            return int(get_list_len(x))
        return X.apply(lam_itemcounter)

# After our transformation vote_count and popularity are even more correlated. 
# It's time to combine them into one feature. I believe taking their average is good enough.
class MeanTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, name):
        self.name = name
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.mean(axis=1).to_frame(self.name)


class PandasFeatureUnion(FeatureUnion):
    def fit_transform(self, X, y=None, **fit_params):
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(
                transformer=trans,
                X=X,
                y=y,
                weight=weight,
                **fit_params)
            for name, trans, weight in self._iter())

        if not result:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs

    def merge_dataframes_by_column(self, Xs):
        return pd.concat(Xs, axis="columns", copy=False)

    def transform(self, X):
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(
                transformer=trans,
                X=X,
                y=None,
                weight=weight)
            for name, trans, weight in self._iter())
        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = self.merge_dataframes_by_column(Xs)
        return Xs
    
def make_union(*transformers, **kwargs):
    n_jobs = kwargs.pop('n_jobs', None)
    verbose = kwargs.pop('verbose', False)
    if kwargs:
        # We do not currently support `transformer_weights` as we may want to
        # change its type spec in make_union
        raise TypeError('Unknown keyword arguments: "{}"'
                        .format(list(kwargs.keys())[0]))
    return PandasFeatureUnion(
        _name_estimators(transformers), n_jobs=n_jobs, verbose=verbose)
def test_missinghome(x):
    return isinstance(x, float)
def test_orilang(x):
    return x == 'en'
def test_multilingual(x):
    return x > 1
def test_released(x):
    return x == 'Released'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_fs", dest="train_fs", required=True, type=str, help="train featureset")
    parser.add_argument("--test_fs", dest="test_fs", required=True, type=str, help="test featureset")
    global FLAGS
    FLAGS, unparsed = parser.parse_known_args()

    data = pd.read_csv(os.path.join(inpath, "tmdb_filter.csv"))
    union = make_union(
        make_sequence(
            FeatureSelector('genres'),
            DictionaryVectorizer('name')
        ),
        make_sequence(
            FeatureSelector('homepage'),
            Binarizer(test_missinghome, 'missing_homepage')
        ),
        make_sequence(
            FeatureSelector('keywords'),
            DictionaryVectorizer('name'),
            TopFeatures(0.5)
        ),
        make_sequence(
            FeatureSelector('original_language'),
            Binarizer(test_orilang, 'en')
        ),
        make_sequence(
            FeatureSelector('production_companies'),
            DictionaryVectorizer('name'),
            TopFeatures(1)
        ),
        make_sequence(
            FeatureSelector('production_countries'),
            DictionaryVectorizer('name'),
            TopFeatures(25)
        ),
        make_sequence(
            FeatureSelector('release_date'),
            DateTransformer()
        ),
        make_sequence(
            FeatureSelector('spoken_languages'),
            ItemCounter(),
            Binarizer(test_multilingual, 'multilingual')
        ),
        make_sequence(
            FeatureSelector('original_language'),
            Binarizer(test_released, 'Released')
        ),    
        make_sequence(
            FeatureSelector('cast'),
            DictionaryVectorizer('name'),
            TopFeatures(0.25),
            SumTransformer('top_cast_count')
        ),
        make_sequence(
            FeatureSelector('crew'),
            DictionaryVectorizer('name', False),
            TopFeatures(1)
        ),
        make_sequence(
            FeatureSelector(['budget', 'runtime', 'vote_average', 'revenue'])
        ),
        make_sequence(
            FeatureSelector(['popularity', 'vote_count']),
            MeanTransformer('popularity_vote')
        )
    )
    union.fit(data)

    data = union.transform(data)
    # removing duplicate columns
    data = data.loc[:,~data.columns.duplicated()]
    
    X_train_T, X_test_T = train_test_split(data)

    
    print(X_train_T.shape)
    print(X_test_T.shape)
    
    
    # Committing features
    authToken = os.getenv("DKUBE_USER_ACCESS_TOKEN")
    # Get client handle
    api = DkubeApi(token=authToken)
    # featureset to use
    fs = [FLAGS.train_fs, FLAGS.test_fs]
    # Commit featuresets
    resp = api.commit_featureset(name=fs[0], df=X_train_T)
    print("train featureset commit response:", resp)
    resp = api.commit_featureset(name=fs[1], df=X_test_T)
    print("test featureset commit response:", resp)