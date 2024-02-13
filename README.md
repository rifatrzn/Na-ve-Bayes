# Na-ve-Bayes

# Overall Goal:
# Work with Naïve Bayes to classify the same dataset 
# Useful python packages:
-For Naïve Bayes,  there is a sklearn package (Recommended):
(https://scikit-learn.org/stable/modules/naive_bayes.html)

-Or there is a nltk package: (https://www.nltk.org/_modules/nltk/classify/naivebayes.html) 

-You may also implement this manually, but this will require extra time and effort, so I leave it up to you.

# Dataset
-The dataset that will be used is the toxic comment dataset. This consists of toxic comments, so be cautioned when viewing the dataset. 

-You can find the training and test sets here: https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data

-The training set (train.csv) will be used for training the LM, while the test set (test.csv) and labels (test_labels.csv) will be used for testing and analyzing your models.

-Take time to understand how the training set and test set are laid out.

# Naive Bayes Model Tasks

-You will implement and test the Naïve Bayes model. This model will predict whether the text is toxic (toxic is 1) or not toxic (toxic is 0).
-Note that your NB model must train on the text itself (CountVectorizer if sklearn) and not use the TF-IDF vectors.


-You will create 2 required functions for NBs (you may create more functions for your own use, but you need to at least create these two as specified):

-train_NB_model(path_to_train_file):
This method trains a naïve bayes model on the training text and returns that trained model.  The format for the train file should follow the same format as the training data file!

-test_NB_model(path_to_test_file, NB_model):
This method tests a trained NB model on some test file and outputs a test file in the same format as the input test file but with 2 columns added: 1) probability of that text being toxic, 2) class prediction (toxic, not toxic).  The format for the input file should follow that of the test file.




