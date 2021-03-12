import pandas as pd
import os
from sklearn.base import BaseEstimator, TransformerMixin

inpath = "/data/merge"
outpath = "/data/filter"

class FeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(self, feature_names):
        self.feature_names = feature_names
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        return X[self.feature_names]
    
if __name__ == "__main__":
    
    data = pd.read_csv(os.path.join(inpath, "tmdb_merge.csv"))

    print(data.shape)
    features = ['genres', 'homepage', 'keywords', 'original_language', 'production_companies', 
               'production_countries', 'release_date', 'spoken_languages', 'original_language',
               'cast', 'crew', 'budget', 'runtime', 'vote_average', 'popularity', 'vote_count', 'revenue']

    filtered_data = FeatureSelector(features).fit_transform(data)
    filtered_data = filtered_data.loc[filtered_data['revenue'] != 0]
    filtered_data['revenue'].dropna(inplace=True)

    print(filtered_data.shape)

    filtered_data.to_csv(os.path.join(outpath, "tmdb_filter.csv"), index=False)