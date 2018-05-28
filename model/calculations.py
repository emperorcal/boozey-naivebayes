import pandas as pd

class Model(object):
    def __init__(self, filePath):
        #Load beer review data into dataframe
        self.filePath = filePath
        self.beerDataframe = pd.read_csv(filePath)

    def splitData(self, testPercentage):
    	#Split original dataframe into test and train dataframes by given proportion
        dataframeSize = self.beerDataframe.shape[0] - 1
        splitIndex = int(dataframeSize * (testPercentage/100))
        print(splitIndex)
        self.testBeerData = self.beerDataframe.iloc[:splitIndex]
        self.trainBeerData = self.beerDataframe.iloc[splitIndex:]
