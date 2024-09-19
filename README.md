<h1>🍏📈📉🍎 Apple Stock Prediction 🍏📈📉🍎 </h1>
<br>
<br>
<br>
The goal of this project is to evaluate LSTM models of varying complexities to find out which best predicts future Apple stock prices. 
<br>
An LSTM model was chosen for this task because of its known ability to work with time series data.
<h2>Prerequisites</h2>
•	‘pandas’
<br>
•	‘random’
<br>
•	‘matplotlib’
<br>
•	‘numpy’
<br>
•	‘ta.momentum’ - used to calculate RSI.
<br>
•	‘keras’
<br>
•	‘tensorflow’
<br>
•	‘scikit-learn’
<h2>Data Sources</h2>
<h4>Apple Data:</h4> 
Kaggle (original source: Yahoo! Finance) https://www.kaggle.com/datasets/dilaraahan/apple-stock-prices
<br>
<h4>Google Data:</h4> 
Kaggle https://www.kaggle.com/datasets/surajjoshi26/google-stock-price2004-2023
<br>
<br>
Both datasets licensed under CC0: Public Domain. 
<h2>Time Frame</h2>
Daily stock data from 2008 to 2021.
<h2>Models</h2>
LSTM Model 1: The purpose of Model 1 was to test with low complexity. Therefore, only one variable was used.
<br>
LSTM Model 2: The purpose of Model 2 was to add complexity using only the Apple data. 
<br>
LSTM Model 3: The purpose of Model 3 was to add complexity using the Apple data and Google data. 
<br>
Each model was trained with 50 epochs. Batch sizes varied. Models 1 and 3 were trained with a batch size of 16 and Model 2 was trained with a batch size of 32. Model 2 benefitted from a larger batch size possibly due to accommodate the increased complexity of the additional variables, or to reduce overfitting.
<h2>Feature Engineering</h2> 
Model 2: 
<br>
• cubeRootVolume, created a standardized volume distribution.
<br>
• RSI. 
<br>
<br>
Model 3: 
<br>
• MACDSignalAAPL, MACDSignalGOOG (Moving Average Convergence/Divergence) captures trends in the stock price. 
<br>
• %KAAPL
<br>
• rollingStd5AAPL, (rolling standard deviation over a 5 day period) and is used to account for price volatility. 
<br>
• openCloseDiffGOOG
<br>
• Date components such as: month, day, and day of week. 
<br>
<h2>Data Preprocessing</h2>
<br>
• There were no missing values in the dataset.
<br>
<br>
• All outliers were deemed to be factual for the stock prices in question.
<br>
<br>
• Feature Selection: Variance Inflation Factor (VIF) was used in feature selection to eliminate multicollinearity and ensure the variables were independent. Recursive Feature Elimination (RFE) was utilized to choose relevant variables.
<br>
<br>
• Scaling: The data was scaled using MinMaxScaler to normalize the feature range. Although both the train and test data were transformed, the model was only fitted to the training data. 
<h2>Hyperparameter Tuning</h2>
• Model 1: 
<br>
Units: 10 to 256
<br>
Dropout Rate: 0.0 to 0.5.
<br>
Learning Rate: 0.0001 to 0.01
<br>
L2 Regularization: 0.0 to 0.1.
<br>
Number of Layers: 1 to 5
<br>
Recurrent Dropout: 0.0 to 0.3.
<br>
Activation: 'sigmoid', 'elu', 'linear', 'tanh'
<br>
<br>
• Model 2:
<br>
Units: 10 to 512
<br>
Dropout Rate: 0.0 to 0.4
<br>
Learning Rate: 0.0005 to 0.1
<br>
L2 Regularization: 0.0 to 0.4
<br>
Number of Layers: 1 to 4
<br>
Recurrent Dropout: 0.0 to 0.5
<br>
Activation: 'sigmoid', 'elu', 'linear', 'tanh'
<br>
<br>
• Model 3:
<br>
Units: 50 to 256
<br>
Dropout Rate: 0.0 to 0.5
<br>
Learning Rate: 0.0 to 0.1
<br>
L2 Regularization: 0.0 to 0.1
<br>
Number of Layers: 1 to 3
<br>
Recurrent Dropout: 0.0 to 0.3
Activation: 'sigmoid', 'elu', 'linear', 'tanh'
<h2>Evaluation Metrics</h2>
The evaluation metrics used were:
<br>
• MSE (Mean Squared Error)
<br>
•	RMSE (Root Mean Squared Error)
<br>
•	R^2 (R Squared)
<h2>Resulting Metrics</h2>
LSTM Model 1 (Using historic Apple 'Close' price to predict future Apple 'Close' price):
<br>
TestMSE: 27.5028
<br>
TestRMSE: 5.2443
<br>
Test R2: 0.9791
<br>
Val MSE: 0.5961
<br>
Val RMSE: 0.7721
<br>
Val R^2: 0.9854
<br>
<br>
LSTM Model 2 (Using multiple Apple variables to predict future Apple 'Close' price):
<br>
TestMSE: 5.4115  
<br>
TestRMSE: 2.3263
<br>
Test R2: 0.9959
<br>
Val MSE: 0.5664
<br>
Val RMSE: 0.7526
<br>
Val R^2: 0.9861
<br>
<br>
<br>
LSTM Model 3 (Using Apple and Google variables to predict future Apple 'Close' price):
 <br>
TestMSE: 0.2274
<br>
TestRMSE: 0.4769
<br>
Test R2: 0.9829
<br>
Val MSE: 0.0028
<br>
Val RMSE: 0.0532
<br>
Val R^2: 0.9931 
<h2>Visualizations</h2>
Line Graph of Apple and Google stock prices to understand the data used.
<br>
Learning Curves to track the convergence of each model.
<br>
Line Graphs of Predicted vs Actual to display model performance over time.
<h2>Findings</h2>
While all models performed well in R^2 and validation metrics, there was overfitting present in Model 1 and Model 2. In Model 3 the overfitting was eliminated. It took longer for the model to converge in Model 3 due to the added complexity of adding Google variables, but this is what helped it to succeed in the long run. Adding Google variables definitely helped. 
<h2>Future Work</h2>
Incorporating more data for the same timeframe, like that of Apple competitor IBM or economic data. Incorporating more variables may further improve the model.
<br>
Creating a model using the stacked ensemble method for improved predictions. Models to be considered include: GRU, CNN-LSTM, MLP and XG Boost. Stacking models can improve overall accuracy and reduce errors.
<br>
Test models using walk forward cross validation for further evaluation. 
<br>
Applying a transformer model or bidirectional LSTM to this data could improve long-term forecasting.
