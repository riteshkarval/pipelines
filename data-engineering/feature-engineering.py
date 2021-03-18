import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
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

class FeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(self, feature_names):
        self.feature_names = feature_names
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        return X[self.feature_names]


def extract_items(list_, key, all_=True):
    sub = lambda x: re.sub(r'[^A-Za-z0-9]', '_', x)
    if all_:
        target = []
        for dict_ in eval(list_):
            target.append(sub(dict_[key].strip()))
        return ' '.join(target)
    elif not eval(list_):
        return 'no_data'
    else:
        return sub(eval(list_)[0][key].strip())

class DictionaryVectorizer(BaseEstimator, TransformerMixin):
    
    def __init__(self, key, all_=True):
        self.key = key
        self.all = all_
    
    def fit(self, X, y=None):
        genres = X.apply(lambda x: extract_items(x, self.key, self.all))
        self.vectorizer = CountVectorizer().fit(genres)        
        self.columns = self.vectorizer.get_feature_names()
        return self
        
    def transform(self, X):
        genres = X.apply(lambda x: extract_items(x, self.key))
        data = self.vectorizer.transform(genres)
        return pd.DataFrame(data.toarray(), columns=self.vectorizer.get_feature_names(), index=X.index)




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


class SumTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, series_name):
        self.series_name = series_name
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        return X.sum(axis=1).to_frame(self.series_name)



class Binarizer(BaseEstimator, TransformerMixin):
    
    def __init__(self, condition, name):
        self.condition = condition
        self.name = name
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.apply(lambda x : int(self.condition(x))).to_frame(self.name)


from datetime import datetime

def get_year(date):
    return datetime.strptime(date, '%Y-%m-%d').year

def get_month(date):
    return datetime.strptime(date, '%Y-%m-%d').strftime('%b')

def get_weekday(date):
    return datetime.strptime(date, '%Y-%m-%d').strftime('%a')

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

class ItemCounter(BaseEstimator, TransformerMixin):
        
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        return X.apply(lambda x: int(get_list_len(x)))


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_fs", dest="train_fs", required=True, type=str, help="train featureset")
    parser.add_argument("--test_fs", dest="test_fs", required=True, type=str, help="test featureset")
    global FLAGS
    FLAGS, unparsed = parser.parse_known_args()

    data = pd.read_csv(os.path.join(inpath, "tmdb_filter.csv"))
    union = make_union(
        make_pipeline(
            FeatureSelector('genres'),
            DictionaryVectorizer('name')
        ),
        make_pipeline(
            FeatureSelector('homepage'),
            Binarizer(lambda x: isinstance(x, float), 'missing_homepage')
        ),
        make_pipeline(
            FeatureSelector('keywords'),
            DictionaryVectorizer('name'),
            TopFeatures(0.5)
        ),
        make_pipeline(
            FeatureSelector('original_language'),
            Binarizer(lambda x: x == 'en', 'en')
        ),
        make_pipeline(
            FeatureSelector('production_companies'),
            DictionaryVectorizer('name'),
            TopFeatures(1)
        ),
        make_pipeline(
            FeatureSelector('production_countries'),
            DictionaryVectorizer('name'),
            TopFeatures(25)
        ),
        make_pipeline(
            FeatureSelector('release_date'),
            DateTransformer()
        ),
        make_pipeline(
            FeatureSelector('spoken_languages'),
            ItemCounter(),
            Binarizer(lambda x: x > 1, 'multilingual')
        ),
        make_pipeline(
            FeatureSelector('original_language'),
            Binarizer(lambda x: x == 'Released', 'Released')
        ),    
        make_pipeline(
            FeatureSelector('cast'),
            DictionaryVectorizer('name'),
            TopFeatures(0.25),
            SumTransformer('top_cast_count')
        ),
        make_pipeline(
            FeatureSelector('crew'),
            DictionaryVectorizer('name', False),
            TopFeatures(1)
        ),
        make_pipeline(
            FeatureSelector(['budget', 'runtime', 'vote_average', 'revenue'])
        ),
        make_pipeline(
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