# Stock-Price-Presdiction-using-AI-ML
Methodology: 1  Data Collection and Preprocessing 2   Feature Engineering 3  Model Selection and Training (Logistic Regression,KNN,Decision Tree) 4  Model Evaluation
Methodology
The methodology used in this program aims to predict stock market price trends using historical stock data and machine learning models. The methodologies used are as follows:
2.1  Data Collection and Preprocessing
•	The program starts by uploading a CSV file containing stock market data, which is read into a pandas DataFrame. The Date column is converted to a datetime format, and relevant columns (e.g., 'Open', 'Close', 'Volume') are converted to numeric values, handling errors by coercing non-numeric values to NaN.
•	Rows with missing data are removed to ensure clean data for model training.
2.2   Feature Engineering
•	New features are created, including the difference between 'Open' and 'Close' prices, 'Low' and 'High' prices, and a target variable indicating if the price goes up the next day (1) or not (0).
•	Volatility, volume, and trend indicators such as ATR (Average True Range), CMF (Chaikin Money Flow), MACD (Moving Average Convergence Divergence), and Darvas Box are calculated to enhance the feature set..
2.3  Model Selection and Training
o	Logistic Regression: A statistical model used for binary classification.
o	K-Nearest Neighbors (KNN): A non-parametric algorithm used for classification by measuring the proximity of data points.
o	Decision Tree: A model that splits data into subsets based on feature 
2.4  Model Evaluation
•	The models are evaluated based on the ROC AUC score, which measures their ability to distinguish between price movements. A confusion matrix is also generated for the Logistic Regression model to assess classification performance.
In conclusion, this methodology combines data preprocessing, feature engineering, and machine learning to predict stock price movements.
