import argparse
import os
from dkube.sdk import *
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

inp_path = "/opt/dkube/input"
out_path = "/opt/dkube/output"

if __name__ == "__main__":

    ########--- Parse for parameters ---########

    parser = argparse.ArgumentParser()
    parser.add_argument("--fs", dest="fs", required=True, type=str, help="featureset")

    global FLAGS
    FLAGS, unparsed = parser.parse_known_args()
    fs = FLAGS.fs
    
    ########--- Read features from input FeatureSet ---########

    # Featureset API
    authToken = os.getenv("DKUBE_USER_ACCESS_TOKEN")
    # Get client handle
    api = DkubeApi(token=authToken)

    # Read features
    feature_df = api.read_featureset(name = fs) 
    Y = feature_df['revenue']
    X = feature_df.drop(['revenue']
    
    ########--- Train ---########

    for_params = dict(n_estimators=np.linspace(10,40,4).astype(int), min_samples_split=(2,3), min_samples_leaf=(1,2,3))
    
    forest_grid = GridSearchCV(RandomForestRegressor(random_state=42), for_params, cv=10)
    
    forest_grid.fit(X, Y)
    
    print(f'Forest:\n\t *best params: {forest_grid.best_params_}\n\t *best score: {forest_grid.best_score_}')