# Data Engineering Pipeline

## Pipeline Run

1. Create a jupyterlab IDE in DKube
2. Launch jypyterlab and open terminal in it.
3. In terminal `cd /home/$USERNAME/workspace`
4. Download pipeline.ipynb by running `wget https://raw.githubusercontent.com/riteshkarvaloc/pipelines/main/data-engineering/pipeline.ipynb -O de-pipeline.pynb` in jupyterlab terminal.
5. Open de-pipeline.ipynb and run al the cells.

## Run Seperate jobs

### Adding repos

#### Code repo
1. Name: tmdb
2. Source URL: https://github.com/riteshkarvaloc/pipelines.git
3. Submit

#### Dataset Repo
1. Name: tmdb-merged
2. Source: None
3. Submit

#### Dataset Repo
1. Name: tmdb-clean
2. Source: None
3. Submit

#### Train and test featuresets
Add Featureset tmdb-train-fs and tmdb-test-fs without featurespec upload. 

#### Merge Step [Job:Preprocessing]:
1. Create a preprocessing run from run tab.
2. Add project **tmdb**
3. Image: `docker.io/ocdr/dkube-datascience-tf-cpu:v2.0.0-3`
4. Script: `cd data-engineering; python merging.py`
5. Output dataset: tmdb-merged, mount point: /data/merge
6. Submit

#### Clean Step [Job:Preprocessing]:
1. Create a preprocessing run from run tab.
2. Add project **tmdb**
3. Image: `docker.io/ocdr/dkube-datascience-tf-cpu:v2.0.0-3`
4. Script: `cd data-engineering; python cleaning.py`
5. Input dataset: tmdb-merged, mount point: /data/merge
6. Output dataset: tmdb-clean, mount point: /data/clean
7. Submit

#### Feature Engineering Step [Job:Preprocessing]
1. Create a preprocessing run from run tab.
2. Add project **tmdb**
3. Image: `docker.io/ocdr/dkube-datascience-tf-cpu:v2.0.0-3`
4. Script: `cd data-engineering; python feature-engineering.py --train_fs tmdb-train-fs --test_fs tmdb-test-fs`
5. Input dataset: tmdb-clean, mount point: /data/clean
6. Output Featureset: tmdb-train-fs, mount point: /data/train_fs
7. Output Featureset: tmdb-test-fs, mount point: /data/test_fs
8. Submit
