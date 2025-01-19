<h1>🍏📈📉🍎 Apple Stock Prediction 🍏📈📉🍎 </h1>
<br>
<h2>Introduction</h2>
The goal of this project is to evaluate LSTM models of varying complexities to find out which best predicts future Apple stock prices. 
<h5>Business Use Case:</h5>
This project demonstrates how advanced models can support businesses in competitive industries by improving forecasting and market trend analysis.
<br>
An LSTM model was chosen for this task because of its known ability to work with time series data.
<h2> How To Use </h2>
<br>
• PowerPoint: The resuts and visualizations are available in the attached PowerPoint file. This summarizes the key findings of the project.
<br>
• Jupyter Notebook: Download then load into environment such as Google Colab.
<br>
• Data: The datasets for this project are available to download directly from the links provided in the "Data Sources" section.
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
TestMSE: 23.285
<br>
TestRMSE: 4.825
<br>
Test R2: 0.982
<br>
Val MSE: 0.202
<br>
Val RMSE: 0.450
<br>
Val R^2: 0.995
<br>
<br>
LSTM Model 2 (Using multiple Apple variables to predict future Apple 'Close' price):
<br>
TestMSE: 5.296  
<br>
TestRMSE: 2.301
<br>
Test R2: 0.996
<br>
Val MSE: 0.176
<br>
Val RMSE: 0.419
<br>
Val R^2: 0.996
<br>
<br>
<br>
LSTM Model 3 (Using Apple and Google variables to predict future Apple 'Close' price):
 <br>
TestMSE: 0.316
<br>
TestRMSE: 0.562
<br>
Test R2: 0.976
<br>
Val MSE: 0.003
<br>
Val RMSE: 0.056
<br>
Val R^2: 0.992 
<h2>Visualizations</h2>
Line Graph of Apple and Google stock prices to understand the data used.
<br>
Learning Curves to track the convergence of each model.
<br>
Line Graphs of Predicted vs Actual to display model performance over time.
<h2>Findings</h2>
While all models performed well in R^2 and validation metrics, there was overfitting present in Model 1 and Model 2. In Model 3 the overfitting was eliminated. It took longer for the model to converge in Model 3 due to the added complexity of adding Google variables, but this is what helped it to succeed in the long run. Adding Google variables definitely helped. 
<h2>Future Work</h2>
<h5>Expanded Data Sources</h5>
Integrating additional data sources, such as Google Trends and news sentiment analysis, to evaluate the impact of public perception on stock price movements. 
<br>
Include data from another major competitor, such as Samsung (despite being traded on the Korean stock exchange), to assess how international competition influences Apple's stock performance. 
<h5>Advanced Model Techniques</h5>
Develop a stacked ensemble method combining methods like GRU, CNN-LSTM, MLP, and XGBoost to improve accuracy and reduce errors.
<br>
Explore the application of transformer models or bidirectional LSTMs for better long-term forecasting. 
<h5>Evaluation Enhancements</h5>
Implement walk-forward cross-validation to assess model performance over sequential time periods, providing a more robust evluation framework.
