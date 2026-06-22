"""
Train a model of your choice out of a premade list of models to classify Real vs. Fake Images

This file ties together the different stages of the project into a full training pipeline -
    * Extracting the features into a feature stream of (image, label) per split
    * Having it go through a preprocessing pipeline 
    * Train a model on the training split
    * Train the hyperparameters of the models via the evaluation split
    * Test the model via the test split

Available Models:
1.  Soft-SVM, specifically 'SVC' (kernal SVM): Allows for training the hyperparameters of C, kernal and gamma.
    The training is done on a subset of the dataset as the cost of training an SVM on ~80k images is massive in terms of storage and time.
    To train on this model the pipeline is required to have the following steps:
    a) dim reduction via 'vec-pca' or 'vec-jl'
    b) batch normalization of the images via 'scale' which are hyperparameters trained on the train split and fitted on the val and test splits



How to Run (Example):  
    #   Chose the classifier (see MODEL_REGISTRY below):
        python train_model.py --model rf

    #   Choose a Preprocessing Pipeline:
        Either a prebuilt one or a fully custom one.
        *   Choose a PreBuilt Pipeline (see PIPELINE_REGISTRY below):
            python train_model.py --pipeline fast
        
        *   Costumize Your Pipeline:
            python train_model.py --pipeline [Comment for claude: ADD the Stracture for the pipeline here, make it based on the stracture of ImagePipeline's creation function]


    #   Choosing Maximum Training Sample Sizes:
        python train_model.py --max-train-samples 10000 
        NOTE: For the maximal sample size do --max-train-samples 0

    #   Choosing Maximum Testing Sample Sizes:
        python train_model.py --max-test-samples 10000
        NOTE: For the maximal sample size do --max-test-samples 0

    #   Choosing Output Directory:
        python train_model.py --output-dir [Comment for claude: Add an example of how the folder needs to be specified (if its just the name of the folder, write a name)]
        NOTE: Defaults to `outputs/`


How to Add a New Model to the Registry Go to 'model_registry.py'

Requirements:
    pip install numpy opencv-python scikit-learn joblib
    (plus this project's ``preprocessing`` package and its dependencies)
"""


from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

# --- Project building blocks (imported, not reimplemented) --------------------
from extract_features import get_feature_stream
from prebuilt_pipelines import PrebuiltPipelines
from preprocessing import ImagePipeline

# --- scikit-learn ------------------------------------------------------------
import joblib


from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# A single seed threaded through every random operation (subsampling, PCA, the
# estimators) so the whole run is reproducible.
RANDOM_STATE = 42

# Human-readable class names, indexed by integer label (0 = real, 1 = fake),
# matching the manifest produced by create_split.py.
CLASS_NAMES = ["real", "fake"]

logger = logging.getLogger("train_model")