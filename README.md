# Summary
A Naïve Bayes classifier build from scratch in Python aimed at predicting either a positive or negative beer review based on beer review text, using natural language processing (NLP).

The overall number of beer reviews in the dataset is 31079, with an 80/20 split for training and testing (24863 train / 6216 test reviews).

AUC of 0.797 was achieved with the classifier versus an AUC of 0.917 using Scikit multi NB.

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
| \_\_init\_\_(file_path, file_path_dataset)      |                                      |
| split_data(test_percentage)                      |                                      |
| clean_text(text)                                 |                                      |
| clean_dataframes()                               |                                      | 
| calculate_constants(dataset)                     |                                      |
| load_dataframes()                                |                                      |
| get_word_count(text, search_word)                |                                      |
| predict_review(review)                           |                                      |
| calculate_word_probabilities()                   |                                      |
| load_word_probabilities()                        |                                      |
| calculate_auc_classifier()                       |                                      |
| calculate_auc_scikit()                           |                                      |
| plot_roc()                                       |                                      |
| negative_word_ranking()                          |                                      |
| positive_word_ranking()                          |                                      |

# Results
ROC results:
<p align="center">
  <img height="400" src="https://i.imgur.com/yg5AvVe.png">
</p>

Top 20 positive and negative words:
[Top 20 -ve and +ve words]

# Areas for improvement
* Improved review cleansing
    * removing words that occur frequently in both positive and negative reviews
    * grouping words that are relevant to each other in the context of beer, e.g. aroma / smell / scent
* Using N-grams
    * Bigrams / trigrams as opposed to unigrams, this would be mean the model would interpret "not good" as a group rather than "not" and "good" separately
