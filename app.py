import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

app = Flask(__name__)

# Load the tokenizer and model
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
final_model = load_model('model.h5')

# Class dictionary
class_dict = {'0': 'World', '1': 'Sports', '2': 'Business', '3': 'Sci/Tech'}

# Maximum length for padding
maxlen = 200

@app.route('/')
def home():
    return render_template('index.html')  # HTML page here

@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file is uploaded
    if 'file' not in request.files:
        return render_template('index.html', prediction_text='No file uploaded. Please try again.')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', prediction_text='No file selected. Please try again.')

    # Save the file to process it
    filepath = os.path.join('uploads', file.filename)
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    file.save(filepath)

    # Read and preprocess the content
    with open(filepath, 'r', encoding='utf-8') as f:
        input_text = f.read().lower()  # Perform your preprocessing (e.g., convert to lowercase)

    # Tokenize and pad the input
    inp_token = tokenizer.texts_to_sequences([input_text])
    inp_token = pad_sequences(inp_token, padding='post', maxlen=maxlen)

    # Get prediction from the model
    out = final_model.predict(inp_token)
    predicted_label = np.argmax(out)
    prediction = class_dict.get(str(predicted_label))

    return render_template('index.html', prediction_text=f'The predicted category is: {prediction}')

if __name__ == "__main__":
    app.run(debug=True)
