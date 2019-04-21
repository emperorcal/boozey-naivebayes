from model.calculations import *

# Filepath of dataset
file_path_dataset = "data/beerReviews.csv"

# Filepath of datafolder
file_path = "data"

# Percentage of dataset for training
test_percentage = 80


# Create the beer model
beer_model = Model(file_path, file_path_dataset)
# Split beer review data into Test and Train based on user input
beer_model.split_data(test_percentage)

print(beer_model.test_beer_reviews.head(5).Text)
# Clean review text within test and train dataframe
beer_model.clean_dataframes()

print(beer_model.test_beer_reviews.head(5).Text)
