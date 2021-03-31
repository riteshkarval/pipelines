# Data Engineering Pipeline

## Pipeline Run

1. Create a jupyterlab IDE in DKube
2. Launch jypyterlab and open terminal in it.
3. In terminal `cd /home/$USERNAME/workspace`
4. Download pipeline.ipynb by running `wget https://raw.githubusercontent.com/riteshkarvaloc/pipelines/main/training/pipeline.ipynb -O train-pipeline.pynb` in jupyterlab terminal.
5. Open train-pipeline.ipynb and run al the cells.

## Run Seperate jobs

### Adding repos

#### Model repo
1. Name: tmdb-model
2. Source: None
3. Submit

#### Training Step [Job:training]:
1. Create a training run from run tab.
2. Add project **tmdb**
3. Framework: Sklearn 
4. Script: `cd training; python training.py --fs tmdb-train-fs`
5. Input featureset: tmdb-train-fs, mount point: /opt/dkube/input
6. Output model: tmdb-model, mount point: /opt/dkube/output
7. Submit

#### Evaluation Step [Job:training]:
1. Create a training run from run tab.
2. Add project **tmdb**
3. Framework: Sklearn 
4. Script: `cd training; python evaluation.py --fs tmdb-test-fs`
5. Input featureset: tmdb-test-fs, mount point: /opt/dkube/input
6. Input Model: tmdb-model, mount point: /opt/dkube/model
7. Submit

#### Create test inference
1. Go to training job lineage and click on model
2. In model page click test inference.
3. Provide name and check transformer. 
4. Fill transformer image: `docker.io/ocdr/d3-datascience-sklearn:v0.23.2-1`
5. Transformer script: `training/transformer.py`
6. Choose CPU and submit. 
