from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd


def train_NB_model(path_to_train_file):
    train_data = pd.read_csv(path_to_train_file)
    train_text = train_data['comment_text']
    train_labels = train_data['toxic']

    # Preprocessing the text
    # ...

    # Training the NB model
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_text)
    nb = MultinomialNB()
    nb.fit(X_train, train_labels)

    return nb, vectorizer


def test_NB_model(path_to_test_file, NB_model, vectorizer):
    test_data = pd.read_csv(path_to_test_file)
    test_text = test_data['comment_text']

    # Preprocessing the text
    # ...

    # Predicting by using the NB model
    X_test = vectorizer.transform(test_text)
    test_data['toxic_probability'] = NB_model.predict_proba(X_test)[:, 1]
    test_data['toxic_prediction'] = NB_model.predict(X_test)

    return test_data

train_file = 'train.csv'
test_file = 'test.csv'

# Training the NB model
nb_model, vectorizer = train_NB_model(train_file)

# Testing the NB model
test_result = test_NB_model(test_file, nb_model, vectorizer)

# Saving the test result to a CSV file
test_result.to_csv('test_result.csv', index=False)


# Loading the test labels
test_labels = pd.read_csv('test_labels.csv')

# Merging the test result with the test labels
test_data = pd.merge(test_result, test_labels, on='id')

# Calculating the accuracy of the NB model
accuracy = (test_data['toxic_prediction'] == test_data['toxic']).mean()
print(f"Accuracy: {accuracy}")

# Spliting the test data into toxic and non-toxic subsets
toxic_test_data = test_data[test_data['toxic'] != -1]
non_toxic_test_data = test_data[test_data['toxic'] == 0]

# Calculating the accuracy on each subset
toxic_accuracy = (toxic_test_data['toxic_prediction'] == toxic_test_data['toxic']).mean()
non_toxic_accuracy = (non_toxic_test_data['toxic_prediction'] == non_toxic_test_data['toxic']).mean()

# Printing out the accuracies

print("Accuracy on toxic subset:", toxic_accuracy)
print("Accuracy on non-toxic subset:", non_toxic_accuracy)
