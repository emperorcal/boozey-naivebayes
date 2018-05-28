import pandas as pd
from model.calculations import *

#Filepath of dataset
filePath = "data/beerReviews.csv"
#Percentage of dataset for training
testPercentage = 80

beerModel = Model(filePath)

beerModel.splitData(testPercentage)
