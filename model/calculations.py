import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from autocorrect import spell
from collections import Counter
import re


class Model(object):
	def __init__(self, file_path, file_path_dataset):
		# Load beer review data into dataframe
		self.file_path = file_path
		self.file_path_dataset = file_path_dataset
		self.beer_reviews = pd.read_csv(file_path_dataset)

		# Define other class variables used in calculations
		self.test_beer_reviews = None
		self.train_beer_reviews = None
		self.words = None
		self.number_of_reviews = None
		self.negative_review_count = None
		self.positive_review_count = None
		self.negative_review_words = None
		self.negative_review_word_count = None
		self.positive_review_words = None
		self.positive_review_word_count = None
		self.probability_negative = None
		self.probability_positive = None

	def split_data(self, test_percentage):
		# Shuffle beer review dataframe
		self.beer_reviews = self.beer_reviews.sample(frac=1)
		# Split beer review dataframe into test and train dataframes by given proportion
		dataframe_size = self.beer_reviews.shape[0] - 1
		split_index = int(dataframe_size * (test_percentage / 100))
		self.test_beer_reviews = self.beer_reviews.iloc[split_index:]
		self.train_beer_reviews = self.beer_reviews.iloc[:split_index]

	def clean_text(self, text):
		# Clean review text, to allow for improved tokenisation
		# Split into words by spaces
		self.words = text.split()
		# Perform spelling correction
		self.words = [spell(word) for word in self.words]
		# Remove all punctuation
		table = str.maketrans('', '', string.punctuation)
		self.words = [word.translate(table) for word in self.words]
		# Remove any non-alphabetic words
		self.words = [word for word in self.words if word.isalpha()]
		# Make all lowercase
		self.words = [word.lower() for word in self.words]
		# Filter out stop words (assuming English language)
		filter_words = set(stopwords.words('english'))
		self.words = [word for word in self.words if word not in filter_words]
		# Lemmatize words to reduce variance
		lemmatizer = WordNetLemmatizer()
		self.words = [lemmatizer.lemmatize(word) for word in self.words]
		# Return one string rather than a list of words
		self.words = ' '.join(self.words)
		return self.words

	def clean_dataframes(self):
		# Go through both dataframes and clan review text iteratively
		print("Cleaning training dataframe...")
		counter_loop = 0
		for index, row in self.train_beer_reviews.iterrows():
			print("{}/{} ({}%)".format(counter_loop + 1, self.train_beer_reviews.shape[0], ((counter_loop + 1) / self.train_beer_reviews.shape[0]) * 100))
			counter_loop = counter_loop + 1
			# Use clean_text() to clean review text
			self.train_beer_reviews.at[index, 'Text'] = self.clean_text(row['Text'])

		self.train_beer_reviews.to_csv(self.file_path + r'\train_beer_reviews_cleaned.csv', index=None, header=True)

		print("Cleaning testing dataframe...")
		counter_loop = 0
		for index, row in self.test_beer_reviews.iterrows():
			print("{}/{} ({}%)".format(counter_loop + 1, self.test_beer_reviews.shape[0], ((counter_loop + 1) / self.test_beer_reviews.shape[0]) * 100))
			counter_loop = counter_loop + 1
			# Use clean_text() to clean review text
			self.test_beer_reviews.at[index, 'Text'] = self.clean_text(row['Text'])

		self.test_beer_reviews.to_csv(self.file_path + r'\test_beer_reviews_cleaned.csv', index=None, header=True)

		return None

	def calculate_constants(self, dataset):
		# Get total number of reviews from training dataframe
		self.number_of_reviews = dataset.shape[0]

		# Calculate number of negative reviews from training dataframe
		self.negative_review_count = dataset.loc[dataset['Category Score'] == -1].shape[0]

		# Calculate number of positive reviews
		self.positive_review_count = dataset.loc[dataset['Category Score'] == 1].shape[0]

		# Group all negative review words and count
		self.negative_review_words = " ".join([row[3].lower() for index, row in dataset.iterrows() if row[0] == -1])
		self.negative_review_word_count = len(self.negative_review_words.split())

		# Group all positive review words and count
		self.positive_review_words = " ".join([row[3].lower() for index, row in dataset.iterrows() if row[0] == 1])
		self.positive_review_word_count = len(self.positive_review_words.split())

		# Probability of a word being negative
		self.probability_negative = self.negative_review_count / self.number_of_reviews

		# Probability of a word being positive
		self.probability_positive = self.positive_review_count / self.number_of_reviews
		return None

	def load_dataframes(self):
		# Load beer reviews with cleansed review text
		self.test_beer_reviews = pd.read_csv(self.file_path + r'\test_beer_reviews_cleaned.csv')
		self.train_beer_reviews = pd.read_csv(self.file_path + r'\train_beer_reviews_cleaned.csv')

		# Calculate constants with training dataset
		self.calculate_constants(self.train_beer_reviews)
		return None

	def get_word_count(self, text, search_word):
		# Split text into words based on whitespace
		words = re.split('\s+', text)
		# Use Counter to get count of specific word within text
		word_count_list = Counter(words)
		word_count = word_count_list.get(search_word, 0)
		return word_count

	def predict_review(self, review):
		# Assign arbitrary base value for multiplication
		probability_negative_review = 1
		probability_positive_review = 1

		# Clean review text
		review = self.clean_text(review)

		# Loop through each word in the review and calculate the likelihood of both being negative and positive review
		for word in review.split():
			probability_word_given_negative = self.get_word_count(self.negative_review_words, word) / self.negative_review_word_count
			probability_word_given_positive = self.get_word_count(self.positive_review_words, word) / self.positive_review_word_count

			# Use Naive Bayes Classifier with additive smoothing (with denominator removed)
			probability_negative_given_word = (1 + (probability_word_given_negative * self.probability_negative))
			probability_positive_given_word = (1 + (probability_word_given_positive * self.probability_positive))

			# Multiply this probability with the other word probabilities within review
			probability_negative_review *= probability_negative_given_word
			probability_positive_review *= probability_positive_given_word

		# Assign a classification based on which probability is greater
		if probability_negative_review > probability_positive_review:
			review_prediction = -1
		else:
			review_prediction = 1
		return review_prediction
