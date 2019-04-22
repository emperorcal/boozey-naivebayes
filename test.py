from model.calculations import *

# Filepath of dataset
file_path_dataset = "data/beerReviews.csv"

# Filepath of datafolder
file_path = "data"

# Percentage of dataset for training
test_percentage = 80

# Create the beer model
beer_model = Model(file_path, file_path_dataset)

# First time ----------
# Split beer review data into Test and Train based on user input
# beer_model.split_data(test_percentage)
# Clean review text within test and train dataframe
# beer_model.clean_dataframes()

# Load cleaned dataframes
beer_model.load_dataframes()

# Load calculated word probabilities
beer_model.calculate_word_probabilities()

#print(len(beer_model.negative_review_words))
#print(len(beer_model.positive_review_words))
beer_model.calculate_auc()