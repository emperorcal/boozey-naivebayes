from model.calculations import *

# Filepath of dataset
file_path_dataset = "data/beerReviews.csv"

# Filepath of datafolder
file_path = "data"

# Create the beer model
beer_model = Model(file_path, file_path_dataset)

# Load cleaned dataframes
beer_model.load_dataframes()

# Load pre-calculated word probabilities
beer_model.load_word_probabilities()

# Calculate AUC of classifier
#beer_model.calculate_auc_classifier()

# Calculate AUC of scikit Multi NB classifier
#beer_model.calculate_auc_scikit()

#Plot ROC curves
#beer_model.plot_roc()
beer_model.negative_wordcloud()
beer_model.positive_wordcloud()