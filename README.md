# Summary
A Naïve Bayes classifier build from scratch in Python aimed at predicting either a positive or negative beer review based on beer review text, using natural language processing (NLP).

The overall number of beer reviews in the dataset is 31079, with an 80/20 split for training and testing (24863 train / 6216 test reviews).

AUC of 0.797 was achieved with the classifier versus an AUC of 0.917 using a Scikit multi NB model.

[Wordclouds](#wordclouds) were created to show the words most likely to produce a positive and negative beer review.

# Theory
Bayes’ law is a ubiquitous mathematical theorem describing the probability of an event, based on prior knowledge of conditions (potentially) related to the event. Given by:
<p align="center">
  <img height="40" src="https://i.imgur.com/Ag0V9up.png">
</p>
We get to the Naïve Bayes classifier by adopting the ‘naïve’ assumption that all of the features are independent. For example, a banana has several ‘features’ – yellow in colour, slightly curved, slippery once peeled. When calculating the probability of an object being a banana, a Naïve Bayes classifier would assume that each feature independently accounts for the probability of being a banana.

Applied to the exam question of predicting if a beer review is positive or negative:
<br>
<p align="center">
  <img height="40" src="https://i.imgur.com/aRyUidS.png">
</p>
<p align="center">
  <img height="40" src="https://i.imgur.com/saecUyD.png">
</p>
The denominator can be removed as the classifier will be comparing both the positive and negative conditional probabilities (as shown above). To calculate the conditional probability of a review, we multiply the conditional probability of each word together (for that class), for example:
<p align="center">
  <img height="65" src="https://i.imgur.com/RUQurUT.png">
</p>
To avoid issues with unseen words in future predictions, Laplace smoothing (also known as additive smoothing) is performed. This is done by simply adding a constant to the numerator within our Naïve Bayes classifier:
<p align="center">
  <img height="65" src="https://i.imgur.com/xkBry0i.png">
</p>
To simplify calculations, the class probability can be brought out to give the final equation:
<p align="center">
  <img height="80" src="https://i.imgur.com/55SA2Jm.png">
</p>
To evaluate the output quality of the classifier, the Receiver Operating Characteristic (ROC) metric is used. Visualised graphically, the ROC curve shows the relationship between the True Positive Rate (TPR) and the False Positive Rate (FPR).
<p align="center">
  <img height="40" src="https://i.imgur.com/J7dBI9K.png">
</p>
<p align="center">
  <img height="40" src="https://i.imgur.com/bJcHu1z.png">
</p>
Area Under the Curve (AUC) is a metric that calculates the area under the ROC curve, a single value of 0 to 1, with 1 being the most desirable. This metric is also used to assess the quality of the classifier.

# Process
Due to the large dataset, the process was broken down into stages to assist with computation time:
<p align="center">
  <img height="165" src="https://i.imgur.com/oOubYpp.png">
</p>

# Usage

## Installation

To install the required Python packages, run the following:

```
pip install -r requirements.txt
```

## Class Methods
Please see [example.py](example.py) as an example on how to use the model.

An exhaustive list of class methods:

| Method                                          | Description                          | 
| :-----------------------------------------------|:------------------------------------| 
| \_\_init\_\_(file_path, file_path_dataset)      | Class constructor requiring file path to store data and beer data filename              |
| split_data(test_percentage)                      | Splits the original dataset into test and train dataframes by test percentage  |
| clean_text(text)                                 | Cleans text as per process above, input and output is a string                |
| clean_dataframes()                               | Cleans all review text in both test and train dataframes and stores both as separate CSV files in the class file path                                    | 
| calculate_constants(dataset)                     | With given dataset, calculates all constants required for Naive Bayes classifier  |
| load_dataframes()                                | Loads cleaned test and train CSV files into dataframes                       |
| get_word_count(text, search_word)                | Given search_word (substring) returns integer of count within given text     |
| predict_review(review)                           | Uses constants and word probabilities within Naive Bayes classifier to predict specific outcome of given review                      |
| calculate_word_probabilities()                   | Calculates both positive and negative word probabilities for all distinct words in train dataframe and then stores as a CSV in class file path                                     |
| load_word_probabilities()                        | Loads word probabilities from CSV to dataframe                                     |
| calculate_auc_classifier()                       | Predicts reviews on the test dataframe using the classifier and returns the AUC  |
| calculate_auc_scikit()                           | Predicts reviews on the test dataframe using Scikit Multi NB classifier and returns the AUC                                     |
| plot_roc()                                       | Plots both AUC values on chart for comparison |
| beer_color_func()                                       |   Returns random beer colour, required for wordcloud                                   |
| negative_wordcloud()                          | Creates beer masked wordcloud of negative words           |
| positive_wordcloud()                          | Creates beer masked wordcloud of positive words  |


# Results

## ROC

The model achieved an AUC of 0.80 which is relatively good considering how basic it is. For comparison, a Scikit Multi Naive Bayes model was created using the same dataset. [Outlined below](#areas-for-improvement) are areas for development to potentially improve the model.
<p align="center">
  <img height="400" src="https://i.imgur.com/yg5AvVe.png">
</p>


## Wordclouds

WordClouds for words that are most likely to give positive and negative reviews:
<p align="center">
  <img src="https://i.imgur.com/vYV6qIA.png" width="430" />
  <img src="https://i.imgur.com/YI5IzIC.png" width="430" />
</p>

# Areas for improvement
* Improved review cleansing
    * splitting words with dashes, currently processed as block, e.g. "after-taste" goes to "aftertaste" after text cleaning
    * removing words that occur frequently in both positive and negative reviews, e.g. beer 
    * grouping words that are relevant to each other in the context of beer, e.g. aroma / smell / scent
* Using N-grams
    * Bigrams / trigrams as opposed to unigrams, this would be mean the model would interpret "not good" as a group rather than "not" and "good" separately
