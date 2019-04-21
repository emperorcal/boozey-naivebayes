import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from autocorrect import spell
from collections import Counter
import re
import plotly.plotly as py
import plotly.graph_objs as go
from sklearn import metrics


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

	def calculate_auc(self):
		actual_scores = []
		predicted_scores = []
		counter = 1

		for index, row in self.test_beer_reviews.iterrows():
			actual_scores.append(row['Category Score'])
			predicted_scores.append(self.predict_review(row['Text']))
			print("{}/{} {}%".format(counter, self.test_beer_reviews.shape[0], (counter / self.test_beer_reviews.shape[0])*100))
			counter = counter + 1

		# Generate the roc curve using scikit-learn.
		fpr, tpr, thresholds = metrics.roc_curve(actual_scores, actual_scores, pos_label=1)

		# Calculate AUC (area under curve) - closer to 1, the "better" the predictions
		auc_value = metrics.auc(fpr, tpr)
		print("AUC of the predictions: {0}".format(metrics.auc(fpr, tpr)))
		return auc_value

	def negative_word_ranking(self):
		# Show the top 25 words that have the highest probability of giving a negative score
		# Create dataframe to store word and score
		score_df = pd.DataFrame(columns=['word', 'score'])

		# Counter to print when looping
		counter = 1

		# To save time, remove duplicated from negative word list
		distinct_negative_words = list(dict.fromkeys(self.negative_review_words.split()))

		# Loop through each negative word, calculate probability and insert into dataframe
		for word in distinct_negative_words:
			# Calculate probability
			probability_word_given_negative = self.get_word_count(self.negative_review_words, word) / self.negative_review_word_count
			probability_negative_given_word = (probability_word_given_negative * self.probability_negative)

			# Store score in dataframe
			temp_df = pd.DataFrame({'word': [word], 'score': [probability_negative_given_word]}, columns=score_df.keys())
			score_df = score_df.append(temp_df)

			# Print progress
			print("{}/{}".format(counter, len(distinct_negative_words)))
			counter = counter + 1

		# Sort by score, remove duplicates and take top 25
		score_df = score_df.sort_values(by=['score'], ascending=False).drop_duplicates(keep='first').head(25)

		# Plot bar chart
		data = [
			go.Bar(
				x=score_df['word'],
				y=score_df['score']
			)
		]

		layout = go.Layout(
			title='Words most likely to produce a negative beer review'
		)

		fig = go.Figure(data=data, layout=layout)

		py.iplot(fig, filename='negative-words-bar-chart')
		return None

	def positive_word_ranking(self):
		# Show the top 25 words that have the highest probability of giving a positive score
		# Create dataframe to store word and score
		score_df = pd.DataFrame(columns=['word', 'score'])

		# Counter to print when looping
		counter = 1

		# To save time, remove duplicated from negative word list
		distinct_positive_words = list(dict.fromkeys(self.positive_review_words.split()))

		# Loop through each negative word, calculate probability and insert into dataframe
		for word in distinct_positive_words:
			# Calculate probability
			probability_word_given_positive = self.get_word_count(self.positive_review_words, word) / self.positive_review_word_count
			probability_positive_given_word = (probability_word_given_positive * self.probability_positive)

			# Store score in dataframe
			temp_df = pd.DataFrame({'word': [word], 'score': [probability_positive_given_word]}, columns=score_df.keys())
			score_df = score_df.append(temp_df)

			# Print progress
			print("{}/{}".format(counter, len(distinct_positive_words)))
			counter = counter + 1

		# Sort by score, remove duplicates and take top 25
		score_df = score_df.sort_values(by=['score'], ascending=False).drop_duplicates(keep='first').head(25)

		# Plot bar chart
		data = [
			go.Bar(
				x=score_df['word'],
				y=score_df['score']
			)
		]

		layout = go.Layout(
			title='Words most likely to produce a positive beer review'
		)

		fig = go.Figure(data=data, layout=layout)

		py.iplot(fig, filename='positive-words-bar-chart')
		return None