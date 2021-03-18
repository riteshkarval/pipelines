import pandas as pd
import os

inpath = "/data/merge"
outpath = "/data/clean"

    
if __name__ == "__main__":
    
    data = pd.read_csv(os.path.join(inpath, "tmdb_merge.csv"))
    
    # Removing rows with NULL/empty values
    print(data.shape)
    filtered_data = data.loc[data['revenue'] != 0]
    filtered_data['revenue'].dropna(inplace=True)
    
    # removing duplicate columns
    filtered_data = filtered_data.loc[:,~filtered_data.columns.duplicated()]

    print(filtered_data.shape)

    filtered_data.to_csv(os.path.join(outpath, "tmdb_filter.csv"), index=False)