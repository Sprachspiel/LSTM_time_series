import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from pandas.plotting import register_matplotlib_converters

%matplotlib inline
%config InlineBackend.figure_format='retina'

register_matplotlib_converters()
sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 22, 10

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

from google.colab import drive
drive.mount("/content/gdrive", force_remount=True)

raw_data = ("/content/gdrive/MyDrive/wind_dataset.csv")
df = pd.read_csv(raw_data,
  parse_dates=['DATE'], 
  index_col="DATE"
)
df.drop(['IND', 'IND.1', 'IND.2'], axis=1, inplace=True)
df['T.MAX'].fillna(df['T.MAX'].mean
(), inplace=True)
df['T.MIN'].fillna(df['T.MIN'].mean
(), inplace=True)
df['T.MIN.G'].fillna(df['T.MIN.G'].mean
(), inplace=True)

df['day_of_month'] = df.index.day
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month

df_by_month = df.resample('M').sum()
