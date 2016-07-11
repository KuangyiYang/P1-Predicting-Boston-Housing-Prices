# Udacity: Machine Learning Nanodegree, Project 1-Predicting-Boston-Housing-Prices
# Project Overview
In this project, you will apply basic machine learning concepts on data collected for housing prices in the Boston, Massachusetts area to predict the selling price of a new home. You will first explore the data to obtain important features and descriptive statistics about the dataset. Next, you will properly split the data into testing and training subsets, and determine a suitable performance metric for this problem. You will then analyze performance graphs for a learning algorithm with varying parameters and training set sizes. This will enable you to pick the optimal model that best generalizes for unseen data. Finally, you will test this optimal model on a new sample and compare the predicted selling price to your statistics.

# Project Description
The Boston housing market is highly competitive, and you want to be the best real estate agent in the area. To compete with your peers, you decide to leverage a few basic machine learning concepts to assist you and a client with finding the best selling price for their home. Luckily, you’ve come across the Boston Housing dataset which contains aggregated data on various features for houses in Greater Boston communities, including the median value of homes for each of those areas. Your task is to build an optimal model based on a statistical analysis with the tools available. This model will then be used to estimate the best selling price for your clients' homes.

# Run Instruction:

```python
from sklearn import datasets

city_data = datasets.load_boston()

# Get the labels and features from the housing data
housing_prices = city_data.target
housing_features = city_data.data

```

# Software and Libraries
- Python 2.7
- NumPy
- scikit-learn
- matplotlib
- iPython Notebook
