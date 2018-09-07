import pandas as pd
from model.calculations import *

#Filepath of dataset
filePath = "data/beerReviews.csv"
#Percentage of dataset for training
testPercentage = 80


#Create the beer model
beerModel = Model(filePath)
#Split beer review data into Test and Train based on user input
beerModel.splitData(testPercentage)
#Display bar chart of Train score distribution
beerModel.displayScore()

print(beerModel.cleanText("This a review of a beer calld Rosebeer. I thank it is rubbsh but it had a gud aroma!!! It's bettr than a fish that is fishin in a fishery."))