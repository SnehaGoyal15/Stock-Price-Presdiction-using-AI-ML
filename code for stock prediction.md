import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
import warnings
warnings.filterwarnings('ignore')

from google.colab import files
uploaded = files.upload()

filename = next(iter(uploaded))
print(f"Uploaded File: {filename}")

df = pd.read_csv(filename)

df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

columns_to_convert = ['Open', 'High', 'Low', 'Last', 'Close', 'VWAP', 'Volume', 'Turnover', 'Deliverable Volume', '%Deliverble']
for col in columns_to_convert:
    df[col] = pd.to_numeric(df[col], errors='coerce')


df.dropna(inplace=True)

df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']

df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

atr_period = 14
high_low = df['High'] - df['Low']
high_close = np.abs(df['High'] - df['Close'].shift(1))
low_close = np.abs(df['Low'] - df['Close'].shift(1))
true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df['ATR'] = true_range.rolling(window=atr_period).mean()

#(CMF)
df['Money Flow Multiplier'] = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
df['Money Flow Volume'] = df['Money Flow Multiplier'] * df['Volume']
df['CMF'] = df['Money Flow Volume'].rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()

#(MACD)
ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema_12 - ema_26

#Darvas Box
def darvas_box(df):
    high, low = [np.nan], [np.nan]
    for i in range(1, len(df) - 1):
        if df['High'].iloc[i] > df['High'].iloc[i-1] and df['High'].iloc[i] > df['High'].iloc[i+1]:
            high.append(df['High'].iloc[i])
        else:
            high.append(np.nan)
        if df['Low'].iloc[i] < df['Low'].iloc[i-1] and df['Low'].iloc[i] < df['Low'].iloc[i+1]:
            low.append(df['Low'].iloc[i])
        else:
            low.append(np.nan)
    high.append(np.nan)
    low.append(np.nan)
    df['Darvas High'] = high
    df['Darvas Low'] = low
    df['Darvas Range'] = df['Darvas High'] - df['Darvas Low']
    return df

df = darvas_box(df)
df.dropna(subset=['open-close', 'low-high', 'ATR', 'CMF', 'MACD', 'Darvas Range'], inplace=True)

plt.figure(figsize=(15,5))
plt.plot(df['Date'], df['Close'])
plt.title(f'Close Price for {filename.split(".")[0]}', fontsize=15)
plt.ylabel('Price in Rupees')
plt.show()

plt.figure(figsize=(15,5))
plt.plot(df['Date'], df['ATR'], label='ATR (Volatility)', color='purple')
plt.title(f'Average True Range (ATR) for {filename.split(".")[0]}', fontsize=15)
plt.ylabel('ATR')
plt.legend()
plt.show()

plt.figure(figsize=(15,5))
plt.plot(df['Date'], df['CMF'], label='Chaikin Money Flow (CMF)', color='orange')
plt.axhline(0, linestyle='--', color='grey', linewidth=1)
plt.title(f'Chaikin Money Flow (CMF) for {filename.split(".")[0]}', fontsize=15)
plt.ylabel('CMF')
plt.legend()
plt.show()

plt.figure(figsize=(15,5))
plt.plot(df['Date'], df['MACD'], label='MACD', color='green')
plt.axhline(0, linestyle='--', color='grey', linewidth=1)
plt.title(f'Moving Average Convergence Divergence (MACD) for {filename.split(".")[0]}', fontsize=15)
plt.ylabel('MACD')
plt.legend()
plt.show()

plt.figure(figsize=(15,5))
plt.plot(df['Date'], df['Darvas Range'], label='Darvas Box Range', color='blue')
plt.title(f'Darvas Box Range for {filename.split(".")[0]}', fontsize=15)
plt.ylabel('Range')
plt.legend()
plt.show()

plt.figure(figsize=(12,10))
sb.heatmap(df.corr(), annot=True, cmap='coolwarm', cbar=True)
plt.title(f'Correlation Heatmap for {filename.split(".")[0]}')
plt.show()

features = df[['open-close', 'low-high', 'ATR', 'CMF', 'MACD', 'Darvas Range']]
target = df['target']

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

X_train, X_valid, Y_train, Y_valid = train_test_split(features_scaled, target, test_size=0.1, random_state=2022)

param_grid_lr = {'C': [0.1, 1, 10, 100]}
grid_lr = GridSearchCV(LogisticRegression(), param_grid_lr, scoring='roc_auc', cv=5)
grid_lr.fit(X_train, Y_train)
best_lr = grid_lr.best_estimator_

param_grid_knn = {'n_neighbors': range(1, 21)}
grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, scoring='roc_auc', cv=5)
grid_knn.fit(X_train, Y_train)
best_knn = grid_knn.best_estimator_

param_grid_dt = {'max_depth': range(1, 21), 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
grid_dt = GridSearchCV(DecisionTreeClassifier(), param_grid_dt, scoring='roc_auc', cv=5)
grid_dt.fit(X_train, Y_train)
best_dt = grid_dt.best_estimator_

ensemble_model = VotingClassifier(estimators=[
    ('lr', best_lr),
    ('knn', best_knn),
    ('dt', best_dt)
], voting='soft')
ensemble_model.fit(X_train, Y_train)

models = [best_lr, best_knn, best_dt, ensemble_model]
for model in models:
    print(f"{model.__class__.__name__}:")
    print("Training Accuracy:", model.score(X_train, Y_train))
    print("Validation Accuracy:", model.score(X_valid, Y_valid))

# Confusion matrix
predictions = ensemble_model.predict(X_valid)
conf_matrix = metrics.confusion_matrix(Y_valid, predictions)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Class 0', 'Class 1'])
disp.plot(cmap='viridis', colorbar=True)
plt.title("Confusion Matrix for Voting Classifier")
plt.show()

