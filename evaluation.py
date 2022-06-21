from .model import *
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from pandas.plotting import register_matplotlib_converters

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend();

y_pred = model.predict(X_test)

y_train_inv = wind_transformer.inverse_transform(y_train.reshape(1, -1))
y_test_inv = wind_transformer.inverse_transform(y_test.reshape(1, -1))
y_pred_inv = wind_transformer.inverse_transform(y_pred)

plt.plot(np.arange(0, len(y_train)), y_train_inv.flatten(), 'g', label="history")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test_inv.flatten(), marker='.', label="true")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_pred_inv.flatten(), 'r', label="prediction")
plt.ylabel('Wind Speed')
plt.xlabel('Time Step')
plt.legend()
plt.show();

plt.plot(y_test_inv.flatten(), marker='.', label="true")
plt.plot(y_pred_inv.flatten(), 'r', label="prediction")
plt.ylabel('Wind Speed')
plt.xlabel('Time Step')
plt.legend()
plt.show();
