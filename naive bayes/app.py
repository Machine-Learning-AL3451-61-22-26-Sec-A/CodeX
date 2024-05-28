import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load inbuilt training data
@st.cache
def load_train_data():
    data = {
        'document': [
            'great', 'amazing', 'awesome', 'terrible', 'horrible',
            'fantastic', 'bad', 'good', 'excellent', 'poor'
        ],
        'label': [
            'positive', 'positive', 'positive', 'negative', 'negative',
            'positive', 'negative', 'positive', 'positive', 'negative'
        ]
    }
    return pd.DataFrame(data)

# Load inbuilt test data
@st.cache
def load_test_data():
    data = {
        'document': [
            'nice', 'bad', 'good', 'awful', 'wonderful',
            'disappointing', 'great', 'horrible', 'fantastic', 'mediocre'
        ],
        'label': [
            'positive', 'negative', 'positive', 'negative', 'positive',
            'negative', 'positive', 'negative', 'positive', 'negative'
        ]
    }
    return pd.DataFrame(data)

# Train the naive Bayesian classifier
def train_classifier(train_data):
    X_train = train_data['document']
    y_train = train_data['label']
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    classifier = MultinomialNB()
    classifier.fit(X_train_vec, y_train)
    return vectorizer, classifier

# Test the classifier and compute accuracy
def test_classifier(test_data, vectorizer, classifier):
    X_test = test_data['document']
    y_test = test_data['label']
    X_test_vec = vectorizer.transform(X_test)
    y_pred = classifier.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Streamlit UI
def main():
    st.title("Naive Bayesian Classifier")
    st.write("This app demonstrates a naive Bayesian classifier using an inbuilt dataset.")

    # Load inbuilt data
    train_df = load_train_data()
    test_df = load_test_data()

    vectorizer, classifier = train_classifier(train_df)
    accuracy = test_classifier(test_df, vectorizer, classifier)

    st.subheader("Classifier Accuracy")
    st.write(f"The accuracy of the classifier on the test dataset is: {accuracy:.2f}")

if __name__ == "__main__":
    main()

