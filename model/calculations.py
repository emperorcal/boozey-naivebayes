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

		# Define other class variables
		self.test_beer_reviews = None
		self.train_beer_reviews = None
		self.words = None

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

	def load_dataframes(self):
		self.test_beer_reviews = pd.read_csv(self.file_path + r'\test_beer_reviews_cleaned.csv')
		self.train_beer_reviews = pd.read_csv(self.file_path + r'\train_beer_reviews_cleaned.csv')
		return None

	def get_word_count(self, text, search_word):
		# Split text into words based on whitespace
		words = re.split('\s+', text)
		# Use Counter to get count of specific word within text
		word_count_list = Counter(words)
		word_count = word_count_list.get(search_word, 0)
		return word_count
	