# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import streamlit as st
from pyngrok import ngrok

# Mount Google Drive to access the dataset in Google Colab
from google.colab import drive
drive.mount('/content/drive')

# Read the uploaded file from Google Drive
df = pd.read_csv('/content/drive/MyDrive/ML_Important_Questions (1).csv')

# Data Preprocessing
print("Null Values:\n", df.isnull().sum())

# Label encoding for the 'Topic' column
le = LabelEncoder()
df['Topic'] = le.fit_transform(df['Topic'])

# TF-IDF Vectorizer to convert 'Question' text into numerical features
tfidf = TfidfVectorizer(max_features=100)
X_text = tfidf.fit_transform(df['Question']).toarray()

# Combining the TF-IDF features with the 'Topic' feature
X = np.concatenate((X_text, df[['Topic']].values), axis=1)

# Target variable 'Marks'
y = df['Marks']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))

# Plot confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.show()

# Feature Importance
feature_importances = model.feature_importances_
feature_names = tfidf.get_feature_names_out().tolist() + ['Topic']

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print(importance_df.head(10))

# Predict Marks for a New Question
new_question = ["Explain Random Forest Algorithm."]
new_q_tfidf = tfidf.transform(new_question).toarray()
topic_encoded = le.transform(['Supervised Learning'])  # Assuming topic is known

new_input = np.concatenate((new_q_tfidf, [topic_encoded]), axis=1)

predicted_marks = model.predict(new_input)
print("Predicted Marks:", predicted_marks[0])

# Save the model and preprocessing tools for future use
joblib.dump(model, 'ml_model.pkl')
joblib.dump(tfidf, 'tfidf.pkl')
joblib.dump(le, 'label_encoder.pkl')

# Create a Streamlit app to interact with the model
st.title("ML PYQ Marks Predictor")

# Load the saved model and preprocessing tools
model = joblib.load('ml_model.pkl')
tfidf = joblib.load('tfidf.pkl')
le = joblib.load('label_encoder.pkl')

# Text input for question
question = st.text_area("Enter your ML Question")

# Dropdown for selecting a topic
topic = st.selectbox("Select Topic", le.classes_)

# Button to trigger prediction
if st.button("Predict Marks"):
    # Process the input question and topic
    question_tfidf = tfidf.transform([question]).toarray()
    topic_encoded = le.transform([topic])
    input_data = np.concatenate((question_tfidf, [topic_encoded]), axis=1)
    
    # Make the prediction
    prediction = model.predict(input_data)
    
    # Display the predicted marks
    st.success(f"Predicted Marks: {prediction[0]}")

# Install Streamlit and pyngrok (For Colab or other environments where you need to install dependencies)
!pip install streamlit
!pip install pyngrok

# Set up ngrok for external access to the Streamlit app
ngrok.set_auth_token("YOUR_AUTHTOKEN")  # Replace with your ngrok authtoken

# Run the Streamlit app in the background using nohup (in case of Colab)
!nohup streamlit run app.py &
