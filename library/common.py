import os
import gc
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import csv
import math
import time
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from sklearn.cluster import MiniBatchKMeans
from imblearn.under_sampling import ClusterCentroids

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree

from tensorflow import keras
from keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Input, BatchNormalization, Lambda, Layer
from keras.utils import plot_model
from keras import layers,regularizers
from tensorflow.keras import backend as K
from matplotlib import pyplot
from library.ae import *
from library.dae import *
from library.sae import *
from library.vae import *


# 醫療領域
# datasets = ['ecoli-0_vs_1', 'ecoli-0-1-3-7_vs_2-6', 'ecoli1', 'ecoli2', 'ecoli3', 'ecoli4', 'haberman', 'new-thyroid1', 
#             'new-thyroid2', 'pima', 'wisconsin', 'yeast-0-5-6-7-9_vs_4', 'yeast1', 'yeast-1_vs_7', 'yeast-1-2-8-9_vs_7',
#             'yeast-1-4-5-8_vs_7', 'yeast-2_vs_4', 'yeast-2_vs_8', 'yeast3', 'yeast4', 'yeast5', 'yeast6']

# baseline<0.7
# datasets = ['glass1','yeast1','haberman','vehicle1','vehicle3','glass-0-1-5_vs_2','yeast-0-3-5-9_vs_7-8','glass-0-1-6_vs_2','glass-0-1-4-6_vs_2','glass2',
#             'yeast-1-4-5-8_vs_7','flare-F','yeast4','yeast-1-2-8-9_vs_7','winequality-red-8_vs_6','abalone-17_vs_7-8-9-10','winequality-white-3_vs_7',
#             'winequality-red-8_vs_6-7','abalone-19_vs_10-11-12-13','winequality-white-3-9_vs_5','poker-8-9_vs_5','poker-8_vs_6','abalone19']

# 44 datasets
datasets = ['glass1', 'ecoli-0_vs_1', 'wisconsin', 'pima', 'iris0', 'glass0', 'yeast1', 'haberman', 'vehicle2', 'vehicle1', 'vehicle3', 
            'glass-0-1-2-3_vs_4-5-6', 'vehicle0', 'ecoli1', 'new-thyroid1', 'new-thyroid2', 'ecoli2', 'segment0', 'glass6', 'yeast3', 
            'ecoli3', 'page-blocks0', 'yeast-2_vs_4', 'yeast-0-5-6-7-9_vs_4', 'vowel0', 'glass-0-1-6_vs_2', 'glass2', 'shuttle-c0-vs-c4', 
            'yeast-1_vs_7', 'glass4', 'ecoli4', 'page-blocks-1-3_vs_4', 'abalone9-18', 'glass-0-1-6_vs_5', 'shuttle-c2-vs-c4', 
            'yeast-1-4-5-8_vs_7', 'glass5', 'yeast-2_vs_8', 'yeast4', 'yeast-1-2-8-9_vs_7', 'yeast5', 'ecoli-0-1-3-7_vs_2-6', 
            'yeast6', 'abalone19']

set_size = 64


def train_ae(ver, size, x_train, x_test):
    if ver == 'ae_210':
        x_train_encoded, x_test_encoded = train_ae_210(x_train, x_test, size)
    elif ver == 'ae_220':
        x_train_encoded, x_test_encoded = train_ae_220(x_train, x_test, size)
    elif ver == 'ae_230':
        x_train_encoded, x_test_encoded = train_ae_230(x_train, x_test, size)
    elif ver == 'ae_240':
        x_train_encoded, x_test_encoded = train_ae_240(x_train, x_test, size)
    elif ver == 'ae_310':
        x_train_encoded, x_test_encoded = train_ae_310(x_train, x_test, size)
    elif ver == 'ae_320': 
        x_train_encoded, x_test_encoded = train_ae_320(x_train, x_test, size)
    elif ver == 'ae_330':
        x_train_encoded, x_test_encoded = train_ae_330(x_train, x_test, size)
    elif ver == 'ae_410':
        x_train_encoded, x_test_encoded = train_ae_410(x_train, x_test, size)
    elif ver == 'ae_420':
        x_train_encoded, x_test_encoded = train_ae_420(x_train, x_test, size)
    elif ver == 'ae_510':
        x_train_encoded, x_test_encoded = train_ae_510(x_train, x_test, size)
    elif ver == 'ae_610':
        x_train_encoded, x_test_encoded = train_ae_610(x_train, x_test, size)
    elif ver == 'ae_710':
        x_train_encoded, x_test_encoded = train_ae_710(x_train, x_test, size)

    if ver == 'dae_210':
        x_train_encoded, x_test_encoded = train_dae_210(x_train, x_test, size)
    elif ver == 'dae_220':
        x_train_encoded, x_test_encoded = train_dae_220(x_train, x_test, size)
    elif ver == 'dae_230':
        x_train_encoded, x_test_encoded = train_dae_230(x_train, x_test, size)
    elif ver == 'dae_240':
        x_train_encoded, x_test_encoded = train_dae_240(x_train, x_test, size)
    elif ver == 'dae_310':
        x_train_encoded, x_test_encoded = train_dae_310(x_train, x_test, size)
    elif ver == 'dae_320': 
        x_train_encoded, x_test_encoded = train_dae_320(x_train, x_test, size)
    elif ver == 'dae_330':
        x_train_encoded, x_test_encoded = train_dae_330(x_train, x_test, size)
    elif ver == 'dae_410':
        x_train_encoded, x_test_encoded = train_dae_410(x_train, x_test, size)
    elif ver == 'dae_420':
        x_train_encoded, x_test_encoded = train_dae_420(x_train, x_test, size)
    elif ver == 'dae_510':
        x_train_encoded, x_test_encoded = train_dae_510(x_train, x_test, size)
    elif ver == 'dae_610':
        x_train_encoded, x_test_encoded = train_dae_610(x_train, x_test, size)
    elif ver == 'dae_710':
        x_train_encoded, x_test_encoded = train_dae_710(x_train, x_test, size)

    if ver == 'sae_210':
        x_train_encoded, x_test_encoded = train_sae_210(x_train, x_test, size)
    elif ver == 'sae_220':
        x_train_encoded, x_test_encoded = train_sae_220(x_train, x_test, size)
    elif ver == 'sae_230':
        x_train_encoded, x_test_encoded = train_sae_230(x_train, x_test, size)
    elif ver == 'sae_240':
        x_train_encoded, x_test_encoded = train_sae_240(x_train, x_test, size)
    elif ver == 'sae_310':
        x_train_encoded, x_test_encoded = train_sae_310(x_train, x_test, size)
    elif ver == 'sae_320': 
        x_train_encoded, x_test_encoded = train_sae_320(x_train, x_test, size)
    elif ver == 'sae_330':
        x_train_encoded, x_test_encoded = train_sae_330(x_train, x_test, size)
    elif ver == 'sae_410':
        x_train_encoded, x_test_encoded = train_sae_410(x_train, x_test, size)
    elif ver == 'sae_420':
        x_train_encoded, x_test_encoded = train_sae_420(x_train, x_test, size)
    elif ver == 'sae_510':
        x_train_encoded, x_test_encoded = train_sae_510(x_train, x_test, size)
    elif ver == 'sae_610':
        x_train_encoded, x_test_encoded = train_sae_610(x_train, x_test, size)
    elif ver == 'sae_710':
        x_train_encoded, x_test_encoded = train_sae_710(x_train, x_test, size)

    if ver == 'vae_210':
        x_train_encoded, x_test_encoded = train_vae_210(x_train, x_test, size)
    elif ver == 'vae_220':
        x_train_encoded, x_test_encoded = train_vae_220(x_train, x_test, size)
    elif ver == 'vae_230':
        x_train_encoded, x_test_encoded = train_vae_230(x_train, x_test, size)
    elif ver == 'vae_240':
        x_train_encoded, x_test_encoded = train_vae_240(x_train, x_test, size)
    elif ver == 'vae_310':
        x_train_encoded, x_test_encoded = train_vae_310(x_train, x_test, size)
    elif ver == 'vae_320': 
        x_train_encoded, x_test_encoded = train_vae_320(x_train, x_test, size)
    elif ver == 'vae_330':
        x_train_encoded, x_test_encoded = train_vae_330(x_train, x_test, size)
    elif ver == 'vae_410':
        x_train_encoded, x_test_encoded = train_vae_410(x_train, x_test, size)
    elif ver == 'vae_420':
        x_train_encoded, x_test_encoded = train_vae_420(x_train, x_test, size)
    elif ver == 'vae_510':
        x_train_encoded, x_test_encoded = train_vae_510(x_train, x_test, size)
    elif ver == 'vae_610':
        x_train_encoded, x_test_encoded = train_vae_610(x_train, x_test, size)
    elif ver == 'vae_710':
        x_train_encoded, x_test_encoded = train_vae_710(x_train, x_test, size)
    return x_train_encoded, x_test_encoded
