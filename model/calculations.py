import pandas as pd
import matplotlib.pyplot as plt
import string
import nltk
from nltk.corpus import stopwords

class Model(object):
    def __init__(self, filePath):
        #Load beer review data into dataframe
        self.filePath = filePath
        self.beerDataframe = pd.read_csv(filePath)

    def splitData(self, testPercentage):
    	#Split beer review dataframe into test and train dataframes by given proportion
        dataframeSize = self.beerDataframe.shape[0] - 1
        splitIndex = int(dataframeSize * (testPercentage/100))
        self.testBeerData = self.beerDataframe.iloc[splitIndex:]
        self.trainBeerData = self.beerDataframe.iloc[:splitIndex]

    def displayScore(self):
    	#Display bar chart of the grouping of scores (as ratio)
    	uniqueScores = self.trainBeerData.Score.unique()
    	for score in uniqueScores:
    		uniqueScoreCount = pd.DataFrame(pd.value_counts(self.trainBeerData.Score.values, sort=score, normalize=True).reset_index())
    		uniqueScoreCount.columns = ['score', 'count']
    		uniqueScoreCount = uniqueScoreCount.sort_values(by='score')
    		break
    	uniqueScoreCount.plot.bar(x='score', y='count', rot=0, legend=False)
    	plt.show()

    def cleanText(self, text):
    	#Clean review text, to allow for improved tokenisation
    	#Split into words by whitespace
    	self.words = text.split()
    	#Remove all punctuation
    	table = str.maketrans('', '', string.punctuation)
    	self.words = [word.translate(table) for word in self.words]
    	#Remove any non-alphabetic words
    	self.words = [word for word in self.words if word.isalpha()]
    	#Make all lowercase
    	self.words = [word.lower() for word in self.words]
    	#Filter out stop words
    	filterWords = set(stopwords.words('english'))
    	self.words = [word for word in self.words if not word in filterWords]
    	return self.words



