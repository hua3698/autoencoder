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
