import pandas as pd
import matplotlib.pyplot as plt

class Model(object):
    def __init__(self, filePath):
        #Load beer review data into dataframe
        self.filePath = filePath
        self.beerDataframe = pd.read_csv(filePath)

    def splitData(self, testPercentage):
    	#Split original dataframe into test and train dataframes by given proportion
        dataframeSize = self.beerDataframe.shape[0] - 1
        splitIndex = int(dataframeSize * (testPercentage/100))
        self.testBeerData = self.beerDataframe.iloc[splitIndex:]
        self.trainBeerData = self.beerDataframe.iloc[:splitIndex]

    def displayScore(self):
    	#Display bar chart of the grouping of scores
    	uniqueScores = self.trainBeerData.Score.unique()
    	for score in uniqueScores:
    		uniqueScoreCount = pd.DataFrame(pd.value_counts(self.trainBeerData.Score.values, sort=score, normalize=True).reset_index())
    		uniqueScoreCount.columns = ['score', 'count']
    		uniqueScoreCount = uniqueScoreCount.sort_values(by='score')
    		break
    	uniqueScoreCount.plot.bar(x='score', y='count', rot=0, legend=False)
    	plt.show()
