<h4>🍏📈📉🍎 Apple Stock Prediction 🍏📈📉🍎 </h4>
<br>
<br>
<br>
The goal of this project is to evaluate several machine learning models to be used in predicting future Apple stock prices. The algorithms/models to be employed are: 
<br>
•	LassoCV
<br>
•	LSTM
<br>
•	RidgeCV
<br>
<br>
<br>
<h3>Prerequisites</h3>
 <br>
•	‘pandas’
<br>
•	‘seaborn’
<br>
•	‘matplotlib’
<br>
•	‘numpy’
<br>
•	‘ta’
<br>
•	‘keras’
<br>
•	‘tensorflow’
<br>
•	‘scikit-learn’
<br>
<br>
<br>
<h3>Data</h3>
<br>
Source: Kaggle (original source: Yahoo! Finance) https://www.kaggle.com/datasets/dilaraahan/apple-stock-prices
<br>
Time Frame: Daily stock prices from 1980 to 2020
<br>
<br>
<br>
<h3>Data Preprocessing</h3>
<br>
There were no missing values in the dataset.
<br>
All outliers were deemed to be factual for the stock prices in question.
<br>
Feature engineering: created important variables such as `averageDailyPrice`, `volumeWeightedPrice`, and `cubeRootVolume`.
<br>
The standard ‘Volume’ variable was skewed in the dataset. Creating ‘cubeRootVolume’ made that data less skewed and gave it more of a standard distribution. Thus, it was used for predictions.
<br>
<br>
<br>
<h3>Exploratory Data Analysis</h3>
<br>
Apple stock prices were near zero until around 2007.
<br>
An analysis of the average daily price from September to November 2007-2020 was conducted. This analysis, while not central to the project, provides insights into historical stock trends around iPhone announcements and launches.
<br>
(insert photo here)
<br>
As mentioned in the preprocessing section, ‘cubeRootVolume’ was created. The traditional ‘Volume’ metric was skewed and converting it to ‘cubeRootVolume’ gave it a more standard distribution.
<br>
(insert photo here)
<br>
<br>
<br>
<h3>Models</h3>
<br>
LSTM (Long Short-Term Memory): Chosen to build two models, one where the future ‘Close’ price is predicted from the historic ‘Close’ price, and one where multiple variables are used to predict the future ‘Close’ price. LSTM is known for its ability to handle seasonality and trends, and is also known for its accuracy with time series forecasting.
<br>
LassoCV: Chosen to build a model where multiple variables are used to predict the future ‘Close’ price. LassoCV helps to prevent overfitting.
<br>
RidgeCV: Chosen to build a model where future ‘Close’ price is predicted from historic ‘Close’ price as L2 regularization ensures the model handles multicollinearity well.
<br>
<br>
<br>
<h3>Evaluation Metrics</h3>
<br>
The evaluation metrics used were:
<br>
•	RMSE (Root Mean Squared Error)
<br>
•	MAE (Mean Absolute Error)
<br>
•	R^2 (R Squared)
<br>
<br>
<h2>Single Variable Metrics</h2>
<br>
RidgeCV:
<br>
TestMSE: 1.43
<br>
TestRMSE: 1.20
<br>
Test R2: 0.999
<br>
Val MSE: 0.04
<br>
Val RMSE: 0.19
<br>
Val R^2: 0.999
<br>
<br>
LSTM: 
<br>
TestMSE: 26.09
<br>
TestRMSE: 5.11
<br>
Test R2: 0.98
<br>
Val MSE: 0.14
<br>
Val RMSE: 0.38
<br>
Val R^2: 0.99
<br>
<br>
<br>
<h2>Multiple Variable Metrics</h2>
<br>
LassoCV:
 <br>
TestMSE: 4.09
<br>
TestRMSE: 2.02
<br>
Test R2: 0.997
<br>
Val MSE: 0.36
<br>
Val RMSE: 0.60
<br>
Val R^2: 0.99 
<br>
<br>
LSTM: 
 <br>
TestMSE: 45.9
<br>
TestRMSE: 6.77
<br>
Test R2: 0.965
<br>
Val MSE: 0.1.89
<br>
Val RMSE: 1.37
<br>
Val R^2: 0.967
<br>
 
(insert learning curve 1)
<br>
(insert learning curve 2)
<br>
(insert learning curve 3)
<br>
(insert learning curve 4)
<br>
<br>
<br>
<h3>Findings</h3>
<br>
The regression models (RidgeCV and LassoCV) far outperform LSTM in predicting stock prices. In the first set of models (single variable) the MSE and RMSE are lower in RidgeCV than in LSTM which indicates that RidgeCV is picking up more of the nuances in the data than LSTM. In the second set of models (multiple variables) the MSE and RMSE are both lower in LassoCV and the RMSE is higher with LassoCV. In short, this means that the LassoCV model makes better predictions than the LSTM model. 
<br>
<br>
<br>
<h3>Future Work</h3>
<br>
Incorporating data from industry competitors like Samsung or Google.
<br>
Creating an ensemble method for improved predictions.  
<br>
Creating a dashboard for the data in Tableau.
<br>
<h3>Acknowledgements</h3>
<br>
OpenAI – for providing access to models like ChatGPT, which came to my aid in crafting this ReadMe. 
