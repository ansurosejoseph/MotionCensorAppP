import json
import pickle
import re
import string
import sys
import os

# Preprocess function
def preprocess(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    return text

# Load the trained classifier and vectorizer with absolute paths
MODEL_PATH = r"C:\Users\mathe\source\repos\MotionCensorApp\MotionCensorApp\genre_classifier.pkl"
VECTORIZER_PATH = r"C:\Users\mathe\source\repos\MotionCensorApp\MotionCensorApp\vectorizer.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    print("Error: Model or vectorizer file not found.")
    sys.exit(1)

with open(MODEL_PATH, "rb") as model_file:
    classifier = pickle.load(model_file)

with open(VECTORIZER_PATH, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Load the transcription JSON
def load_transcription(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    segments = data.get("segments", [])
    full_text = " ".join([segment["text"] for segment in segments])
    return full_text

# Predict genre
def predict_genre(json_path):
    try:
        text = load_transcription(json_path)
        processed_text = preprocess(text)
        print(f"Processed Text: '{processed_text}'")

        text_vector = vectorizer.transform([processed_text])
        print(f"Text Vector Shape: {text_vector.shape}")

        # Get prediction probabilities
        genre_probs = classifier.predict_proba(text_vector)
        print(f"Prediction Probabilities: {genre_probs}")

        # Get the predicted genre
        genre = classifier.predict(text_vector)[0]
        print(f"Predicted Genre: {genre}")

        # Print the genre classes and their corresponding probabilities
        print("Genre Classes:", classifier.classes_)
        for cls, prob in zip(classifier.classes_, genre_probs[0]):
            print(f"{cls}: {prob:.4f}")

    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_genre.py <transcription_json_path>")
        sys.exit(1)

    json_path = sys.argv[1]
    predict_genre(json_path)
