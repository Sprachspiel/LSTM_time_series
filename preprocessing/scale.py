from .data import *


train_size = int(len(df) * 0.9)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
print(len(train), len(test))

from sklearn.preprocessing import RobustScaler

f_columns = ['RAIN', 'T.MAX', 'T.MIN', 'T.MIN.G']

f_transformer = RobustScaler()
wind_transformer = RobustScaler()

f_transformer = f_transformer.fit(train[f_columns].to_numpy())
wind_transformer = wind_transformer.fit(train[['WIND']])

train.loc[:, f_columns] = f_transformer.transform(train[f_columns].to_numpy())
train['WIND'] = wind_transformer.transform(train[['WIND']])

test.loc[:, f_columns] = f_transformer.transform(test[f_columns].to_numpy())
test['WIND'] = wind_transformer.transform(test[['WIND']])

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)        
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)
  
time_steps = 10


X_train, y_train = create_dataset(train, train.WIND, time_steps)
X_test, y_test = create_dataset(test, test.WIND, time_steps)
