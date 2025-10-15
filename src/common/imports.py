"""Common imports for Arctic zero-curtain pipeline."""

# Standard library
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from datetime import datetime
from datetime import datetime, timedelta
from datetime import timedelta
from osgeo import gdal, osr
from packaging import version
from pathlib import Path
from scipy.spatial import cKDTree
from shapely.strtree import STRtree
from sklearn.metrics import (classification_report, confusion_matrix,
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc as sk_auc
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.regularizers import l2
from zero_curtain_pipeline.modeling.fixed_code import analyze_feature_importance_fixed
import cartopy.feature as cfeature
import glob
import json
import logging
import os
import os,sys
import pathlib
import pickle
import plotly.express as px
import re
import sklearn.preprocessing
import sys
import threading
import warnings

# Third-party libraries
from IPython.display import HTML
from IPython.display import clear_output
from IPython.display import display
from IPython.display import display, HTML
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from collections import Counter
from contextlib import contextmanager
from folium.plugins import HeatMap, MarkerCluster
from functools import partial
from holoviews import opts
from joblib import Parallel, delayed
from keras import layers
from keras_tuner.tuners import BayesianOptimization
from matplotlib import cm
from matplotlib import gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm
from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.colors import PowerNorm
from matplotlib.colors import PowerNorm, Normalize, LogNorm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib_venn import venn2
from mpl_toolkits.mplot3d import Axes3D
from plotly.subplots import make_subplots
from pyproj import Proj
from rasterio.plot import show
from scipy import stats
from scipy import stats as scipy_stats
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
from scipy.spatial import QhullError
from scipy.stats import gaussian_kde
from scipy.stats import moran
from scipy.stats import pearsonr
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from shapely.geometry import Point
from shapely.geometry import Point, box
from shapely.validation import make_valid
from sklearn.cluster import DBSCAN
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from sklearn.utils import shuffle
from tabulate import tabulate
from tensorflow.keras.callbacks import (
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Layer
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.layers import ConvLSTM2D, Layer
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Input, Dense, Conv1D, BatchNormalization, Dropout
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, BatchNormalization
from tensorflow.keras.layers import Layer, BatchNormalization
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Add
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Add, ConvLSTM2D
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Add, Lambda
from tensorflow.keras.layers import Reshape, Concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from tensorflow.keras.utils import plot_model
from tensorflow.python.framework.errors_impl import NotFoundError
from tqdm import tqdm
from tqdm.auto import tqdm
from tqdm.keras import TqdmCallback
from tqdm.notebook import tqdm
from typing import Dict, List, Tuple, Optional, Set
from typing import Dict, Optional, List
from webdriver_manager.chrome import ChromeDriverManager
from zero_curtain_pipeline.modeling.fixed_code import analyze_spatial_performance_fixed
from zero_curtain_pipeline.modeling.fixed_code import build_improved_zero_curtain_model_fixed
from zero_curtain_standalone import build_zero_curtain_model, BatchNorm5D
import argparse
import calendar
import cartopy.crs as ccrs
import cmasher as cmr
import cmocean
import contextily as ctx
import csv
import ctypes
import dask
import dask.dataframe as dd
import dataframe_image as dfi
import folium
import gc
import geodatasets
import geopandas as gpd
import graphviz as gv
import hnswlib
import holoviews as hv
import keras
import keras_tuner as kt
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import multiprocessing
import numpy as np
import pandas as pd
import platform
import plotly.graph_objects as go
import polars as pl
import psutil
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.feather as pf
import pyarrow.parquet as pq
import random
import rasterio
import scipy.interpolate as interpolate
import scipy.stats as stats
import seaborn as sns
import shutil
import sklearn.experimental
import sklearn.impute
import sklearn.linear_model
import subprocess
import tables
import tensorflow as tf
import time
import tqdm
import traceback
import xarray as xr

