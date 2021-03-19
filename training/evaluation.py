import argparse
import os
from dkube.sdk import *
import numpy as np
import joblib
from sklearn.metrics import r2_score
import mlflow

model_path = "/opt/dkube/model"
fs_path = "/opt/dkube/input"

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
    X = feature_df.drop(['revenue'], axis=1)
    
    ########--- Evaluation ---########
    
    model = joblib.load(os.path.join(model_path, 'model.joblib')) 
    predicted = model.predict(X)

    r2score = r2_score(Y, predicted)
    print(f'Random Forest test score: {r2score}')    
    mlflow.log_metric("R2-Score", r2score)