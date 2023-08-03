BigMart SALE PRICE PREDICTOR

The dataset was collected from kaggle : https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data?select=Train.csv



AIM:
This project aims to predict the Price of an Items by taking the Items (Food, Groceries, Health and Hygine, etc), outlet type and size and other parameters.



DESCRIPTION

What this project Does?
This project takes the parameters from the available items,outlet type and size and other features.
It then predicts the possible price of an Item. 



HOW THIS PROJECT DOES?

The data was collected from https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data?select=Train.csv . 
The data was cleaned (Cleaned_BigMart.csv).
Then LinearRegression model was built on top of it which had 0.488 r2_score.
This project was given the form of a website built on Flask where the Linear Regression model was used to perform predictions.



REQUIREMENTS

click
Flask
Flask-Cors
joblib
numpy
pandas
pickle
python
scikit-learn
sklearn
seaborn
matplotlib


