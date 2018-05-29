import pandas as pd
from model.calculations import *

#Filepath of dataset
filePath = "data/beerReviews.csv"
#Percentage of dataset for training
testPercentage = 80

beerModel = Model(filePath)

beerModel.splitData(testPercentage)

beerModel.displayScore()

print(beerModel.cleanText("This a review of a beer called Rosebeer. I think it is rubbish but it had a good aroma!!! What a sh*t beer. Cock."))