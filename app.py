from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import joblib
import numpy as np
import docx
import PyPDF2
import logging
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load ML tools
try:
    model = joblib.load('ml_model.pkl')
    tfidf = joblib.load('tfidf.pkl')
    le = joblib.load('label_encoder.pkl')
except Exception as e:
    logging.error(f"Error loading ML tools: {e}")
    raise

app = Flask(__name__)

# Temporary folder for file uploads
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/predict', methods=['POST'])
def predict():
    # Check if file is in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    filename = secure_filename(file.filename)

    # Validate file extension
    if not (filename.endswith('.pdf') or filename.endswith('.docx')):
        return jsonify({'error': 'Unsupported file format. Only PDF and DOCX are allowed.'}), 400

    # Save the file temporarily
    try:
        with tempfile.NamedTemporaryFile(delete=False, dir=UPLOAD_FOLDER) as temp_file:
            temp_filepath = temp_file.name
            file.save(temp_filepath)

        # Extract text from the file
        if filename.endswith('.pdf'):
            try:
                with open(temp_filepath, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ''.join(page.extract_text() for page in reader.pages if page.extract_text())
            except Exception as e:
                logging.error(f"Error reading PDF file: {e}")
                return jsonify({'error': 'Failed to process PDF file'}), 500
        elif filename.endswith('.docx'):
            try:
                doc = docx.Document(temp_filepath)
                text = '\n'.join([para.text for para in doc.paragraphs if para.text.strip()])
            except Exception as e:
                logging.error(f"Error reading DOCX file: {e}")
                return jsonify({'error': 'Failed to process DOCX file'}), 500

        # Split text into questions (line-based)
        questions = [line.strip() for line in text.split('\n') if line.strip()]
        predictions = []

        # Process each question
        for q in questions:
            try:
                # Transform question using tfidf
                tfidf_vec = tfidf.transform([q]).toarray()

                # Encode topic (replace with dynamic logic if needed)
                topic_encoded = le.transform(['Supervised Learning'])  # Example topic
                topic_encoded = np.array(topic_encoded).reshape(1, -1)

                # Concatenate tfidf vector and topic encoding
                input_data = np.concatenate((tfidf_vec, topic_encoded), axis=1)

                # Predict using the model
                mark = model.predict(input_data)[0]
                if mark >= 5:  # Threshold for valid questions
                    predictions.append(q)
            except Exception as e:
                logging.warning(f"Error processing question: {q}. Skipping. Error: {e}")
                continue

        return jsonify({'prediction': predictions})

    except Exception as e:
        logging.error(f"Error processing file: {e}")
        return jsonify({'error': 'An error occurred while processing the file'}), 500

    finally:
        # Clean up temporary file
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)

if __name__ == '__main__':
    app.run(debug=True)