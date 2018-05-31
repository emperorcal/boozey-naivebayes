import pandas as pd
from model.calculations import *

#Filepath of dataset
filePath = "data/beerReviews.csv"
#Percentage of dataset for training
testPercentage = 80

beerModel = Model(filePath)

beerModel.splitData(testPercentage)

beerModel.displayScore()

print(beerModel.cleanText("This a review of a beer calld Rosebeer. I thank it is rubbsh but it had a gud aroma!!! It's bettr than a fish that is fishin in a fishery."))