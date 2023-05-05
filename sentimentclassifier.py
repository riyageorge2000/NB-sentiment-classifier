import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
import pickle

#Read our dataset using read_csv()
review = pd.read_csv('reviews.csv')
review = review.rename(columns={'text': 'review'})

#split data
X = review.review
y = review.polarity
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=1)

#fit the vectorizer on the training data
vector = CountVectorizer(stop_words='english', lowercase=False)
vector.fit(X_train)
X_transformed = vector.transform(X_train)
X_test_transformed = vector.transform(X_test)

#Train the model
naivebayes = MultinomialNB()
naivebayes.fit(X_transformed, y_train)

#Save the model
saved_model = pickle.dumps(naivebayes)

#Load the saved model
s = pickle.loads(saved_model)

#Define the Streamlit app
st.header('Sentiment Classifier')
input_text = st.text_area("Please enter the text", value="")
if st.button("Check"):
    input_text_transformed = vector.transform([input_text]).toarray()
    prediction = s.predict(input_text_transformed)
    prediction_mapping = {0: 'NEGATIVE', 1: 'POSITIVE'}
    result = prediction_mapping[prediction[0]]
    st.write(f"Predicted category: {result}")