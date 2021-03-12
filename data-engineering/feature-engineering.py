import pandas as pd
import os
from fit_transform import *
from sklearn.pipeline import make_pipeline, make_union
from sklearn.model_selection import train_test_split

inpath = "/data/filter"
test_fs_path = "/data/test_fs"
train_fs_path = "/data/train_fs"


if __name__ == "__main__":
    data = pd.read_csv(os.path.join(inpath, "tmdb_filter.csv"))
    X_train, X_test, y_train, y_test = train_test_split(data.drop(['revenue'], axis=1), data['revenue']) 
    union = make_union(
        make_pipeline(FeatureSelector('genres'), DictionaryVectorizer('name')),

        make_pipeline(FeatureSelector('homepage'), Binarizer(lambda x: isinstance(x, float), 'missing_homepage')),

        make_pipeline(FeatureSelector('keywords'), DictionaryVectorizer('name'), TopFeatures(0.5)),

        make_pipeline(FeatureSelector('original_language'), Binarizer(lambda x: x == 'en', 'en')),

        make_pipeline(FeatureSelector('production_companies'), DictionaryVectorizer('name'), TopFeatures(1)),

        make_pipeline(FeatureSelector('production_countries'), DictionaryVectorizer('name'), TopFeatures(25)),

        make_pipeline(FeatureSelector('release_date'), DateTransformer()),

        make_pipeline(FeatureSelector('spoken_languages'), ItemCounter(),Binarizer(lambda x: x > 1, 'multilingual')),

        make_pipeline(FeatureSelector('original_language'), Binarizer(lambda x: x == 'Released', 'Released')),    

        make_pipeline(FeatureSelector('cast'), DictionaryVectorizer('name'), TopFeatures(0.25), SumTransformer('top_cast_count')),

        make_pipeline(FeatureSelector('crew'), DictionaryVectorizer('name', False), TopFeatures(1)),

        make_pipeline(FeatureSelector(['budget', 'runtime', 'vote_average'])),

        make_pipeline(FeatureSelector(['popularity', 'vote_count']), MeanTransformer('popularity_vote'))
        )
    
    union.fit(X_train)

    X_train_T = union.transform(X_train)
    X_test_T = union.transform(X_test)

    print(X_train_T.shape)
    print(X_test_T.shape)