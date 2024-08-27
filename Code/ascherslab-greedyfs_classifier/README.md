## Introduction
- You can use this repository for running Machine Learning (with K-fold Cross Validation) and Greedy Feature Selection on classification data via `tenfold_classifier.py` and `greedy_classifier.py`, respectively.
- Please check the scripts for further details.

## Scripts
**greedy_classifier.py** : Python script for greedy feature selection.
**tenfold_classifier.py** : Python script for tenfold cross validation and it is required for running greedy_classifier.py.

**train.csv** : an example set for training
**test.csv** : an example set for blind-test
**stopped_total_feature.csv,stopped_total_feature_w_blind.csv** : an example result CSV for resuming functionality

## Be aware..
- Depends on algorithm/feature, some of features give **warning** message during very early stage of greedy feature selection.
 
## Available algorithms and scoring metrics:
- algorithm : [GB,XGBOOST,RF,ExtraTrees,GAUSSIAN,ADABOOST,KNN,SVC,MLP,DecisionTree,LG]
- scoring metrics: ['roc_auc','matthew','bacc','f1']

## How To Install

1. You can install a conda environment using either `requirements.yaml` or installation commands as follows:  
  
	`conda env create -f requirements.yaml`

	or

	```
	conda create -n greedyFS python=3.10

	conda install -c anaconda scikit-learn=1.2.0
	conda install -c conda-forge pandas tqdm xgboost scikit-learn-intelex
	``` 
  
2. Activate your new conda envrionement :
  
	`source activate greedyFS`  