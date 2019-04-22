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
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import csv


class Model(object):
	def __init__(self, file_path, file_path_dataset):
		# Load beer review data into dataframe
		self.file_path = file_path
		self.file_path_dataset = file_path_dataset
		self.beer_reviews = pd.read_csv(file_path_dataset)

		# Define other class variables used in calculations
		self.test_beer_reviews = None
		self.train_beer_reviews = None
		self.word_probabilities = None
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

		# Probability of a review being negative
		self.probability_review_negative = self.negative_review_count / self.number_of_reviews

		# Probability of a review being positive
		self.probability_review_positive = self.positive_review_count / self.number_of_reviews
		print("Probability word positive: {}, probability word negative:{}".format(self.probability_review_positive, self.probability_review_negative))
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
			# Lookup word in calculated probabilities dataframe
			temp_df = self.word_probabilities.loc[self.word_probabilities['word'] == word]
			# If word exists in train dataframe
			if temp_df.shape[0] != 0:
				probability_word_given_positive = temp_df.iloc[0]['positive_probability']
				probability_word_given_negative = temp_df.iloc[0]['negative_probability']

				# Apply additive smoothing
				probability_negative_given_word = 1 + probability_word_given_negative
				probability_positive_given_word = 1 + probability_word_given_positive

				print(" -ve: {}".format(probability_negative_given_word))
				print(" +ve: {}".format(probability_positive_given_word))

				# Multiply this probability with the other word probabilities within review
				probability_negative_review *= probability_negative_given_word
				probability_positive_review *= probability_positive_given_word

		# Multiply class probability
		probability_negative_review *= self.probability_review_negative
		probability_positive_review *= self.probability_review_positive

		# Assign a classification based on which probability is greater
		if probability_negative_review > probability_positive_review:
			review_prediction = -1
		elif probability_negative_review == probability_positive_review:
			review_prediction = 0
		else:
			review_prediction = 1

		print("'{}'-ve vs. '{}'+ve, score:{}".format(probability_negative_review, probability_positive_review, review_prediction ))
		return review_prediction

	def calculate_word_probabilities(self):
		# Calculate the probability of positive and negative scores for each word in the train dataframe
		# Create dataframe to store all train words and calculated probabilities
		self.word_probabilities = pd.DataFrame(columns=['word', 'positive_probability', 'negative_probability'])

		# Get all train review words
		review_words = self.negative_review_words + self.positive_review_words

		# To save time, remove duplicated words
		distinct_review_words = list(dict.fromkeys(review_words.split()))

		# Counter
		counter = 1

		for word in distinct_review_words:
			# Calculate negative probability
			probability_word_given_negative = self.get_word_count(self.negative_review_words, word) / self.negative_review_word_count

			# Calculate positive probability
			probability_word_given_positive = self.get_word_count(self.positive_review_words, word) / self.positive_review_word_count

			# Store word and probabilities in dataframe
			temp_df = pd.DataFrame({'word': [word], 'positive_probability': [probability_word_given_positive],
					'negative_probability': [probability_word_given_negative]}, columns=self.word_probabilities.keys())
			self.word_probabilities = self.word_probabilities.append(temp_df)

			# Print progress
			print("{} / {} {}%".format(counter, len(distinct_review_words), (counter / len(distinct_review_words))*100))
			counter = counter + 1

		# Write probabilities to CSV for future use
		self.word_probabilities.to_csv(self.file_path + r'\word_probabilities.csv', index=None, header=True)

		return None

	def load_word_probabilities(self):
		# Load training word probabilities into dataframe
		self.word_probabilities = pd.read_csv(self.file_path + r'\word_probabilities.csv')
		return None

	def calculate_auc(self):
		actual_scores = []
		predicted_scores = []
		counter = 1

		for index, row in self.test_beer_reviews.iterrows():
			actual_scores.append(row['Category Score'])
			predicted_scores.append(self.predict_review(row['Text']))
			print("{}/{} {}%".format(counter, self.test_beer_reviews.shape[0], (counter / self.test_beer_reviews.shape[0])*100))
			counter = counter + 1

		# Merge scores and save to csv
		merged_scores = tuple(zip(actual_scores, predicted_scores))

		with open(self.file_path + r'\results.csv', 'w', newline='') as csvFile:
			writer = csv.writer(csvFile)
			writer.writerows(merged_scores)
		csvFile.close()

		# Generate the roc curve using scikit-learn.
		fpr, tpr, thresholds = metrics.roc_curve(actual_scores, predicted_scores, pos_label=1)

		# Calculate AUC (area under curve) - closer to 1, the "better" the predictions
		auc_value = metrics.auc(fpr, tpr)
		print("AUC of the Classifier: {0}".format(metrics.auc(fpr, tpr)))

		# Calculate AUC using Scikit
		vectorizer = CountVectorizer(stop_words='english')
		train_features = vectorizer.fit_transform([row['Text'] for index, row in self.train_beer_reviews.iterrows()])
		test_features = vectorizer.transform([row['Text'] for index, row in self.test_beer_reviews.iterrows()])

		nb = MultinomialNB()
		nb.fit(train_features, [int(row['Category Score']) for index, row in self.train_beer_reviews.iterrows()])

		predictions = nb.predict(test_features)

		fpr, tpr, thresholds = metrics.roc_curve(actual_scores, predictions,
		                                         pos_label=1)
		print("Scikit Multinomial naive bayes AUC: {0}".format(metrics.auc(fpr, tpr)))
		return auc_value

	def negative_word_ranking(self):
		# Show the top 25 words that have the highest probability of giving a negative score
		# Sort by score, remove duplicates and take top 25
		score_df = self.word_probabilities.sort_values(by=['negative_probability'], ascending=False).drop_duplicates(keep='first').head(25)

		# Plot bar chart
		data = [
			go.Bar(
				x=score_df['word'],
				y=score_df['negative_probability']
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
		# Sort by score, remove duplicates and take top 25
		score_df = self.word_probabilities.sort_values(by=['positive_probability'], ascending=False).drop_duplicates(keep='first').head(25)

		# Plot bar chart
		data = [
			go.Bar(
				x=score_df['word'],
				y=score_df['positive_probability']
			)
		]

		layout = go.Layout(
			title='Words most likely to produce a positive beer review'
		)

		fig = go.Figure(data=data, layout=layout)

		py.iplot(fig, filename='positive-words-bar-chart')
		return None