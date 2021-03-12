import pandas as pd
import os

outpath = "/data/merge"

if __name__ == "__main__":
    
    credits = pd.read_csv('https://storage.googleapis.com/tmdb/tmdb_5000_credits.csv', index_col='movie_id')
    movies = pd.read_csv('https://storage.googleapis.com/tmdb/tmdb_5000_movies.csv', index_col='id')

    data = pd.merge(movies, credits)
    print(data.shape)

    data.to_csv(os.path.join(outpath, "tmdb_merge.csv"), index=False)