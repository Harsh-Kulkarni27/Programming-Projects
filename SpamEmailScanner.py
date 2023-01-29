import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the data
data = pd.read_csv('emails.csv')

# Split the data into training and test sets
train_data, test_data, train_labels, test_labels = train_test_split(data['text'], data['label'], test_size=0.2)

# Create a bag-of-words representation of the data
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
train_vectors = vectorizer.fit_transform(train_data)
test_vectors = vectorizer.transform(test_data)

# Train the Random Forest classifier
classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(train_vectors, train_labels)

# Make predictions on the test data
predictions = classifier.predict(test_vectors)

# Print the accuracy and confusion matrix
print(f'Accuracy: {accuracy_score(test_labels, predictions)}')
print(f'Confusion Matrix: \n{confusion_matrix(test_labels, predictions)}')

# Train the SVM classifier
classifier = SVC(kernel='linear', C=1)
classifier.fit(train_vectors, train_labels)

# Make predictions on the test data
predictions = classifier.predict(test_vectors)

# Print the accuracy and confusion matrix
print(f'Accuracy: {accuracy_score(test_labels, predictions)}')
print(f'Confusion Matrix: \n{confusion_matrix(test_labels, predictions)}')


# This script uses the emails.csv file, which should contain two columns: 'text' and 'label'.
# The 'text' column should contain the text of the emails, 
# and the 'label' column should contain the label ('spam' or 'not spam') for each email.
# Also, it uses CountVectorizer to convert text data into numerical data
# and then trains the classifiers (Random Forest and SVM) on that data and then make predictions on test data. 
# The accuracy and confusion matrix are then printed to check the performance of the classifiers.