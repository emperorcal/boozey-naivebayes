import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from autocorrect import spell
from collections import Counter
import re
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import csv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from wordcloud import WordCloud
import random

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
		self.fpr_classifier = None
		self.tpr_classifier = None
		self.thresholds_classifier = None
		self.auc_value_classifier = None
		self.fpr_scikit = None
		self.tpr_scikit = None
		self.thresholds_scikit = None
		self.auc_value_scikit = None

	def split_data(self, test_percentage):
		# Shuffle beer review dataframe
		self.beer_reviews = self.beer_reviews.sample(frac=1)

		# Split beer review dataframe into test and train dataframes by given proportion (as percentage)
		# Calculate index number to split at given proportion
		dataframe_size = self.beer_reviews.shape[0] - 1
		split_index = int(dataframe_size * (test_percentage / 100))

		# Split dataframe into test and train based on calculated index value
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
		# Go through both dataframes and clan review text iteratively, saing to CSV
		# Inform user of progress
		print("Cleaning training dataframe...")

		counter = 0
		for index, row in self.train_beer_reviews.iterrows():
			print("{}/{} ({}%)".format(counter + 1, self.train_beer_reviews.shape[0], ((counter + 1) / self.train_beer_reviews.shape[0]) * 100))
			counter = counter+ 1
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
		# Calculate constants from the input dataset required for the Naive Bayes classifier
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

		# Inform user of Naive Bayes constants based on train dataframe
		print("Probability review positive: {}, probability review negative:{}".format(self.probability_review_positive, self.probability_review_negative))
		print("Number of reviews: {}, Negative review count:{}, Positive review count:{}".format(self.number_of_reviews, self.negative_review_count, self.positive_review_count))
		print("Negative review word count: {}, Positive review word count:{}".format(self.negative_review_word_count, self.positive_review_word_count))
		return None

	def load_dataframes(self):
		# Load test and train dataframes from CSV
		# Load beer reviews with cleansed review text
		self.test_beer_reviews = pd.read_csv(self.file_path + r'\test_beer_reviews_cleaned.csv')
		self.train_beer_reviews = pd.read_csv(self.file_path + r'\train_beer_reviews_cleaned.csv')

		# Calculate constants with training dataset to be used when predicting reviews
		self.calculate_constants(self.train_beer_reviews)
		return None

	def get_word_count(self, text, search_word):
		# Get count of substring within text
		# Split text into words based on whitespace
		words = re.split('\s+', text)
		# Use Counter to get count of specific word within text
		word_count_list = Counter(words)
		word_count = word_count_list.get(search_word, 0)
		return word_count

	def predict_review(self, review):
		# Take review text and return prediction using Naive Bayes classifier
		# Assign arbitrary base value for multiplication
		probability_negative_review = 1
		probability_positive_review = 1

		# Loop through each word in the review and calculate the likelihood of being negative and positive review
		for word in review.split():
			# Lookup word in calculated probabilities dataframe
			temp_df = self.word_probabilities.loc[self.word_probabilities['word'] == word]
			# If word exists in train dataframe
			if temp_df.shape[0] != 0:
				# Store positive and negative probability of word
				probability_word_given_positive = temp_df.iloc[0]['positive_probability']
				probability_word_given_negative = temp_df.iloc[0]['negative_probability']

				# Apply additive smoothing (in case a word has zero probability)
				probability_negative_given_word = 1 + probability_word_given_negative
				probability_positive_given_word = 1 + probability_word_given_positive

				# Multiply positive and negative word probability with the other word probabilities within review
				probability_negative_review *= probability_negative_given_word
				probability_positive_review *= probability_positive_given_word

		# Multiply by class probability
		probability_negative_review *= self.probability_review_negative
		probability_positive_review *= self.probability_review_positive

		# Assign a classification based on which probability is greater
		if probability_negative_review > probability_positive_review:
			review_prediction = -1
		elif probability_negative_review == probability_positive_review:
			review_prediction = 0
		else:
			review_prediction = 1
		
		return review_prediction

	def calculate_word_probabilities(self):
		# Calculate the probability of positive and negative scores for each word in the train dataframe
		# Create dataframe to store all train words and calculated probabilities
		self.word_probabilities = pd.DataFrame(columns=['word', 'positive_probability', 'negative_probability'])

		# Get all train review words
		review_words = self.negative_review_words + self.positive_review_words

		# Inform user of count of total words in train dataframe
		print("Count of total words: {}".format(len(review_words)))

		# To save time, remove duplicated words
		distinct_review_words = list(dict.fromkeys(review_words.split()))

		# Inform user of count of total distinct words in train dataframe
		print("Count of total distinct words: {}".format(len(distinct_review_words)))

		# Counter to display progress
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

	def calculate_auc_classifier(self):
		# Calculate AUC of classifier
		# Create lists to be used by scikit to calculate AUC
		actual_scores = []
		predicted_scores = []
		counter = 1

		# Loop through test dataframe and store actual and predicted into separate lists
		for index, row in self.test_beer_reviews.iterrows():
			actual_scores.append(row['Category Score'])

			# Predict score and append to list
			predicted_score = self.predict_review(row['Text'])
			predicted_scores.append(predicted_score)

			# Print progress
			print("{}/{} {}%".format(counter, self.test_beer_reviews.shape[0], (counter / self.test_beer_reviews.shape[0])*100))
			counter = counter + 1

		# Merge actual and predicted into list
		merged_scores = tuple(zip(actual_scores, predicted_scores))

		# Set headers for CSV
		headers = ['actual','predicted']

		# Save merged list into CSV
		with open(self.file_path + r'\results.csv', 'w', newline='') as csvFile:
			writer = csv.writer(csvFile)
			writer.writerow(headers)
			writer.writerows(merged_scores)
		csvFile.close()

		# Generate the roc curve using scikit-learn.
		self.fpr_classifier, self.tpr_classifier, self.thresholds_classifier = metrics.roc_curve(actual_scores, predicted_scores, pos_label=1)

		# Calculate AUC (area under curve) - closer to 1, the "better" the predictions
		self.auc_value_classifier = metrics.auc(self.fpr_classifier, self.tpr_classifier)
		print("AUC of the Classifier: {0}".format(self.auc_value_classifier))
		return self.auc_value_classifier

	def calculate_auc_scikit(self):
		# Calculate AUC using Scikit
		# Use Vecotrizer to split words into features
		vectorizer = CountVectorizer(stop_words='english')
		train_features = vectorizer.fit_transform([row['Text'] for index, row in self.train_beer_reviews.iterrows()])
		test_features = vectorizer.transform([row['Text'] for index, row in self.test_beer_reviews.iterrows()])

		# Create Multinomial Naive Bayes model and fit train data
		nb = MultinomialNB()
		nb.fit(train_features, [int(row['Category Score']) for index, row in self.train_beer_reviews.iterrows()])

		# Make predictions on test data
		predictions = nb.predict(test_features)

		# Get actual scores from Test dataframe
		actual_scores = self.test_beer_reviews['Category Score'].tolist()

		# Generate the roc curve using scikit-learn.
		self.fpr_scikit, self.tpr_scikit, self.thresholds_scikit = metrics.roc_curve(actual_scores, predictions, pos_label=1)

		# Calculate AUC (area under curve) - closer to 1, the "better" the predictions
		self.auc_value_scikit = metrics.auc(self.fpr_scikit, self.tpr_scikit)
		print("Scikit Multinomial naive bayes AUC: {0}".format(self.auc_value_scikit))
		return self.auc_value_scikit

	def plot_roc(self):
		# Plot ROC for the classifier and the scikit Multi NB
		# Format title
		plt.title('Receiver Operating Characteristic - Classifier vs. Scikit Multi NB')

		# Plot classifier characteristics
		plt.plot(self.fpr_classifier, self.tpr_classifier, 'b', label = 'Classifier AUC = %0.2f' % self.auc_value_classifier)

		# Plot scikit multi NB characteristics
		plt.plot(self.fpr_scikit, self.tpr_scikit, 'g', label = 'Scikit AUC = %0.2f' % self.auc_value_scikit)

		# Format plot
		plt.legend(loc = 'lower right')
		plt.plot([0, 1], [0, 1],'r--')
		plt.xlim([0, 1])
		plt.ylim([0, 1])
		plt.ylabel('True Positive Rate')
		plt.xlabel('False Positive Rate')

		# Display plot
		plt.show()
		return None

	def beer_color_func(self, word, font_size, position, orientation, random_state=None, **kwargs):
		# Required for WordCloud to give random word colour, changing the 'lightness' randomly
		return "hsl(31, 58%%, %d%%)" % random.randint(30, 75)

	def negative_wordcloud(self):
		# Display wordcloud of top 200 words by negative probability
		# Calculate words that occur in both positive and negative reviews
		overlapping_words = list(set(self.negative_review_words.split()).intersection(self.positive_review_words.split()))

		# Create temporary dataframe with overlapping words removed, ranked in order of negative probability and top 200 only
		temp_df = self.word_probabilities[~self.word_probabilities.word.isin(overlapping_words)].sort_values(by=['negative_probability'], ascending=False).head(200)

		# Create dictionary of word and corresponding negative probability required for WordCloud
		negative_word_dict = temp_df.set_index('word')['negative_probability'].to_dict()

		# Grab beer image and transform into NumPy array to be used as a mask for the WordCloud
		beer_mask = np.array(Image.open("beer_mask.png"))

		# Create WordCloud object using the beer mask
		wc = WordCloud(background_color="white", max_words=1000, mask=beer_mask)

		# Generate WordCloud using negative word dictionary, with custom colouring
		wc.generate_from_frequencies(negative_word_dict)
		plt.imshow(wc.recolor(color_func=self.beer_color_func, random_state=3), interpolation="bilinear")

		# Add title, remove axis and plot WordCloud
		plt.title("Words most likely to give a negative beer review")
		plt.axis("off")
		plt.show()
		return None

	def positive_wordcloud(self):
		# Display wordcloud of top 200 words by positive probability
		# Calculate words that occur in both positive and negative reviews
		overlapping_words = list(set(self.negative_review_words.split()).intersection(self.positive_review_words.split()))

		# Create temporary dataframe with overlapping words removed, ranked in order of positive probability and top 200 only
		temp_df = self.word_probabilities[~self.word_probabilities.word.isin(overlapping_words)].sort_values(by=['positive_probability'], ascending=False).head(200)

		# Create dictionary of word and corresponding positive probability required for WordCloud
		positive_word_dict = temp_df.set_index('word')['positive_probability'].to_dict()

		# Grab beer image and transform into NumPy array to be used as a mask for the WordCloud
		beer_mask = np.array(Image.open("beer_mask.png"))

		# Create WordCloud object using the beer mask
		wc = WordCloud(background_color="white", max_words=1000, mask=beer_mask)

		# Generate WordCloud using negative word dictionary, with custom colouring
		wc.generate_from_frequencies(positive_word_dict)
		plt.imshow(wc.recolor(color_func=self.beer_color_func, random_state=3), interpolation="bilinear")

		# Add title, remove axis and plot WordCloud
		plt.title("Words most likely to give a positive beer review")
		plt.axis("off")
		plt.show()
		return None