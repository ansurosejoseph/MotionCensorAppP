import string
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import pickle
from collections import Counter

# Load dataset with balanced genres
def load_dataset():
    data = [
        # Action
        {"text": "explosions and car chases with a lot of action scenes", "genre": "Action"},
        {"text": "hero fights off dozens of enemies in a thrilling sequence", "genre": "Action"},
        {"text": "a high-speed chase with helicopters and police cars", "genre": "Action"},
        {"text": "martial arts combat with intense fight scenes", "genre": "Action"},
        {"text": "the hero saves the world from a deadly threat", "genre": "Action"},
        {"text": "gunfights and hand-to-hand combat in a war zone", "genre": "Action"},
        {"text": "a dangerous mission to rescue hostages", "genre": "Action"},
        {"text": "an undercover agent infiltrates a criminal organization", "genre": "Action"},
        {"text": "the protagonist defuses a bomb just in time", "genre": "Action"},
        {"text": "a battle against an army of mercenaries", "genre": "Action"},

        # Romance
        {"text": "romantic scenes with heartfelt dialogues", "genre": "Romance"},
        {"text": "a couple falling in love against all odds", "genre": "Romance"},
        {"text": "a love story set in a picturesque village", "genre": "Romance"},
        {"text": "an emotional reunion between long-lost lovers", "genre": "Romance"},
        {"text": "a heartwarming wedding scene with vows of love", "genre": "Romance"},
        {"text": "a romantic evening under the stars", "genre": "Romance"},
        {"text": "letters exchanged between two people in love", "genre": "Romance"},
        {"text": "a secret admirer leaves love notes", "genre": "Romance"},
        {"text": "a couple's journey through life's challenges", "genre": "Romance"},
        {"text": "a story of love lost and found again", "genre": "Romance"},

        # Sci-Fi
        {"text": "spaceships and aliens in a futuristic setting", "genre": "Sci-Fi"},
        {"text": "robots and advanced technology dominate the world", "genre": "Sci-Fi"},
        {"text": "humans colonize a distant planet", "genre": "Sci-Fi"},
        {"text": "time travel to prevent a future disaster", "genre": "Sci-Fi"},
        {"text": "cyborgs and artificial intelligence evolve", "genre": "Sci-Fi"},
        {"text": "a dystopian world ruled by machines", "genre": "Sci-Fi"},
        {"text": "virtual reality becomes indistinguishable from real life", "genre": "Sci-Fi"},
        {"text": "an alien invasion threatens Earth", "genre": "Sci-Fi"},
        {"text": "a scientist discovers a parallel universe", "genre": "Sci-Fi"},
        {"text": "a space mission to explore a black hole", "genre": "Sci-Fi"},

        # Comedy
        {"text": "humorous dialogues and funny situations", "genre": "Comedy"},
        {"text": "a hilarious series of misunderstandings", "genre": "Comedy"},
        {"text": "pranks and jokes that leave everyone laughing", "genre": "Comedy"},
        {"text": "a clumsy character who always gets into trouble", "genre": "Comedy"},
        {"text": "a stand-up comedian delivering witty jokes", "genre": "Comedy"},
        {"text": "a funny road trip filled with unexpected events", "genre": "Comedy"},
        {"text": "light-hearted humor in everyday situations", "genre": "Comedy"},
        {"text": "a comedy show full of slapstick humor", "genre": "Comedy"},
        {"text": "two friends get into ridiculous adventures", "genre": "Comedy"},
        {"text": "a comedy of errors at a wedding", "genre": "Comedy"},

        # Crime
        {"text": "gangsters and street fights with lots of violence", "genre": "Crime"},
        {"text": "underworld dealings and brutal gang wars", "genre": "Crime"},
        {"text": "mafia bosses plotting a heist in the city", "genre": "Crime"},
        {"text": "brutal gang revenge in the dark alleys", "genre": "Crime"},
        {"text": "street thugs fighting for territory", "genre": "Crime"},
        {"text": "illegal betting and gangster shootouts", "genre": "Crime"},
        {"text": "detectives investigating a high-profile murder", "genre": "Crime"},
        {"text": "a bank robbery planned by a notorious gang", "genre": "Crime"},
        {"text": "smuggling operations run by dangerous criminals", "genre": "Crime"},
        {"text": "a corrupt cop working with the mafia", "genre": "Crime"},

        # Thriller
        {"text": "dark scenes with a mysterious murder investigation", "genre": "Thriller"},
        {"text": "psychological suspense with unexpected twists", "genre": "Thriller"},
        {"text": "a race against time to stop a serial killer", "genre": "Thriller"},
        {"text": "a conspiracy that threatens national security", "genre": "Thriller"},
        {"text": "a detective uncovers a dangerous secret", "genre": "Thriller"},
        {"text": "a journalist investigates a dangerous cover-up", "genre": "Thriller"},
        {"text": "a missing person case turns into a nightmare", "genre": "Thriller"},
        {"text": "an investigator trapped in a deadly plot", "genre": "Thriller"},
        {"text": "a suspenseful game of cat and mouse", "genre": "Thriller"},
        {"text": "a thrilling adventure with life-or-death stakes", "genre": "Thriller"},

        # Fantasy
        {"text": "a magical world filled with wizards and dragons", "genre": "Fantasy"},
        {"text": "a quest to destroy an evil artifact", "genre": "Fantasy"},
        {"text": "an ancient prophecy that must be fulfilled", "genre": "Fantasy"},
        {"text": "a kingdom ruled by a benevolent sorcerer", "genre": "Fantasy"},
        {"text": "a hero armed with a legendary sword", "genre": "Fantasy"},
        {"text": "mystical creatures and enchanted forests", "genre": "Fantasy"},
        {"text": "a battle between good and evil forces", "genre": "Fantasy"},
        {"text": "a princess with magical powers", "genre": "Fantasy"},
        {"text": "a cursed land awaiting liberation", "genre": "Fantasy"},
        {"text": "a secret portal to a mystical realm", "genre": "Fantasy"},

        # Horror
        {"text": "a haunted house filled with ghosts", "genre": "Horror"},
        {"text": "a terrifying encounter with a monster", "genre": "Horror"},
        {"text": "a group of friends trapped in a creepy forest", "genre": "Horror"},
        {"text": "a cursed object that brings doom", "genre": "Horror"},
        {"text": "a supernatural force terrorizes a town", "genre": "Horror"},
        {"text": "nightmares that become reality", "genre": "Horror"},
        {"text": "a chilling story of possession and exorcism", "genre": "Horror"},
        {"text": "a vampire preying on unsuspecting victims", "genre": "Horror"},
        {"text": "a demon haunting a family", "genre": "Horror"},
        {"text": "a zombie outbreak causing chaos", "genre": "Horror"}
    ]
    texts = [item["text"] for item in data]
    labels = [item["genre"] for item in data]
    return texts, labels



# Preprocess the text
def preprocess(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    return text

# Load and preprocess the dataset
texts, labels = load_dataset()
texts = [preprocess(text) for text in texts]

# Check dataset distribution
print("Dataset Distribution:", Counter(labels))

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
y = labels

# Train-test split (90% train, 10% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Train Multinomial Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

# Evaluate the classifier
y_pred = classifier.predict(X_test)
print("Accuracy on Test Set:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=1))

# Save the trained model and vectorizer
with open("genre_classifier.pkl", "wb") as model_file:
    pickle.dump(classifier, model_file)

with open("vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model and vectorizer saved successfully.")
