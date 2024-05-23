**## 🍏📈📉🍎 Apple Stock Prediction 🍏📈📉🍎 **
<br>
<br>
The goal of this project is to evaluate several machine learning models to be used in predicting future Apple stock prices. The algorithms/models to be employed are: 
•	LassoCV
•	LSTM
•	RidgeCV
<br>
**##Prerequisites:**
 
•	‘pandas’
•	‘seaborn’
•	‘matplotlib’
•	‘numpy’
•	‘ta’
•	‘keras’
•	‘tensorflow’
•	‘scikit-learn’
 

**##Data: **
- Source: Kaggle (original source: Yahoo! Finance) https://www.kaggle.com/datasets/dilaraahan/apple-stock-prices
- Time Frame: Daily stock prices from 1980 to 2020
**Data PreProcessing:**
There were no missing values in the dataset. 
All outliers were deemed to be factual for the stock prices in question. 
Feature engineering: created important variables such as `averageDailyPrice`, `volumeWeightedPrice`, and `cubeRootVolume`.
The standard ‘Volume’ variable was skewed in the dataset. Creating ‘cubeRootVolume’ made that data less skewed and gave it more of a standard distribution. Thus, it was used for predictions. 
**##Exploratory Data Analysis:**
Apple stock prices were near zero until around 2007. 
An analysis of the average daily price from September to November 2007-2020 was conducted. This analysis, while not central to the project, provides insights into historical stock trends around iPhone announcements and launches.
(insert photo here)
As mentioned in the preprocessing section, ‘cubeRootVolume’ was created. The traditional ‘Volume’ metric was skewed and converting it to ‘cubeRootVolume’ gave it a more standard distribution. 
(insert photo here)
**##Models:**
LSTM (Long Short-Term Memory): Chosen to build two models, one where the future ‘Close’ price is predicted from the historic ‘Close’ price, and one where multiple variables are used to predict the future ‘Close’ price. LSTM is known for its ability to handle seasonality and trends, and is also known for its accuracy with time series forecasting.
LassoCV: Chosen to build a model where multiple variables are used to predict the future ‘Close’ price. LassoCV helps to prevent overfitting. 
RidgeCV: Chosen to build a model where future ‘Close’ price is predicted from historic ‘Close’ price as L2 regularization ensures the model handles multicollinearity well. 
**##Evaluation Metrics:**
The evaluation metrics used were: 
•	RMSE (Root Mean Squared Error)
•	MAE (Mean Absolute Error)
•	R^2 (R Squared)
Single Variable Metrics
RidgeCV:
 
TestMSE: 1.43
TestRMSE: 1.20
Test R2: 0.999
Val MSE: 0.04
Val RMSE: 0.19
Val R^2: 0.999
 
LSTM: 
 
TestMSE: 26.09
TestRMSE: 5.11
Test R2: 0.98
Val MSE: 0.14
Val RMSE: 0.38
Val R^2: 0.99
 

Multiple Variable Metrics
LassoCV:
 
TestMSE: 4.09
TestRMSE: 2.02
Test R2: 0.997
Val MSE: 0.36
Val RMSE: 0.60
Val R^2: 0.99 
LSTM: 
 
TestMSE: 45.9
TestRMSE: 6.77
Test R2: 0.965
Val MSE: 0.1.89
Val RMSE: 1.37
Val R^2: 0.967
 
(insert learning curve 1)
(insert learning curve 2)
(insert learning curve 3)
(insert learning curve 4)
**##Findings**
The regression models (RidgeCV and LassoCV) far outperform LSTM in predicting stock prices. In the first set of models (single variable) the MSE and RMSE are lower in RidgeCV than in LSTM which indicates that RidgeCV is picking up more of the nuances in the data than LSTM. In the second set of models (multiple variables) the MSE and RMSE are both lower in LassoCV and the RMSE is higher with LassoCV. In short, this means that the LassoCV model makes better predictions than the LSTM model. 
**##Future Work**
Incorporating data from industry competitors like Samsung or Google.
Creating an ensemble method for improved predictions.  
Creating a dashboard for the data in Tableau.
**##Acknowledgements**
OpenAI – for providing access to models like ChatGPT, which came to my aid in crafting this ReadMe. 
