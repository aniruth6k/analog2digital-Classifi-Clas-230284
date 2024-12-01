# Document Classification Web Application

This repository, **analog2digital-Classifi-Clas-230284**, contains a Flask-based web application that classifies documents into categories using deep learning model. Users can upload files, and the application predicts the category of the document.

## Features

- **Document Upload:** Users can upload files through a user-friendly web interface.
- **Text Classification:** Utilizes a deep learning model to classify text into predefined categories.
- **Responsive UI:** Modern and responsive interface with animations for better user experience.

## Requirements

Before running the application, ensure you have the following installed:

- Python 3.7 or higher
- TensorFlow 2.x
- Flask
- NumPy
- Pickle

## Installation

1. **Clone the Repository:**
    
    ```bash
    git clone https://github.com/your-username/analog2digital-Classifi-Clas-230284.git
    cd analog2digital-Classifi-Clas-230284
    ```
    
2. **Create a Virtual Environment:**
    
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    ```
    
3. **Install Dependencies:**
    
    ```bash
    pip install -r requirements.txt
    ```
    
4. **Prepare Model and Tokenizer:**
    
    - Ensure `model.h5` and `tokenizer.pickle` files are in the root directory of the project. These files are necessary for prediction.

## Running the Application

1. **Start the Flask Server:**
    
    ```bash
    python app.py
    ```
    
2. **Access the Application:** Open your web browser and go to:
    
    ```
    http://127.0.0.1:5000/
    ```
    

## Usage

1. **Upload a Document:**
    
    - Select a file using the file input on the homepage.
    - Click the "Upload" button.
2. **View Results:**
    
    - The application will process the document and display the predicted category.

## File Structure

```
analog2digital-Classifi-Clas-230284/
├── app.py                  # Main Flask application
├── tokenizer.pickle        # Tokenizer for text preprocessing
├── model.h5                # Trained Keras model
├── index.html              # Frontend HTML template
└── uploads/                # Directory to save uploaded files
```

## Model Details

- **Architecture:** Deep learning model (e.g., LSTM or CNN) trained to classify text into one of the following categories:
    - **World**
    - **Sports**
    - **Business**
    - **Sci/Tech**
- **Input:** Tokenized and padded text sequences with a maximum length of 200 tokens.

## Customization

- **Adding New Categories:** Update the `class_dict` in `app.py` to include new classes.
- **Model Training:** Retrain the model with new data and replace `model.h5` with the updated model.
