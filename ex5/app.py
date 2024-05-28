import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Function to train naive Bayesian classifier and compute accuracy, precision, and recall
def train_and_evaluate_classifier(train_data):
    try:
        X_train, y_train = train_data['document'], train_data['label']
    except KeyError as e:
        st.error(f"Error: {e}. Make sure your dataset contains columns named 'document' and 'label'.")
        return None, None, None
    
    # Vectorize the documents using Bag of Words approach
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    
    # Train the classifier
    nb_classifier = MultinomialNB()
    nb_classifier.fit(X_train_vec, y_train)
    
    # Predict on training data
    y_pred = nb_classifier.predict(X_train_vec)
    
    # Compute evaluation metrics
    accuracy = accuracy_score(y_train, y_pred)
    precision = precision_score(y_train, y_pred, average='weighted')
    recall = recall_score(y_train, y_pred, average='weighted')
    
    return accuracy, precision, recall

# Main Streamlit app
def main():
    st.title("Naive Bayesian Classifier for Document Classification")
    
    # Upload training dataset
    st.sidebar.header("Upload Dataset")
    st.sidebar.subheader("Training Dataset")
    uploaded_train_file = st.sidebar.file_uploader("Upload training dataset (CSV)", type=["csv"])

    if uploaded_train_file:
        # Read dataset
        train_df = pd.read_csv(uploaded_train_file)

        # Train and evaluate the classifier
        accuracy, precision, recall = train_and_evaluate_classifier(train_df)

        # Display evaluation metrics
        st.subheader("Evaluation Metrics")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write(f"Precision: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")

if __name__ == "__main__":
    main()


