import random
import json 
bot_name = "Gen G"
import json
import random
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
# Load the intents data
with open("intents.json", "r") as file:
    intents = json.load(file)
 
# Preprocess the data
lemmatizer = WordNetLemmatizer()
stop_words = set(nltk.corpus.stopwords.words("english"))
 
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
    return " ".join(tokens)
 
patterns = []
responses = []
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(preprocess_text(pattern))
        responses.append(intent["responses"])
 
# Vectorize the patterns
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(patterns)
 
# Run the chatbot
 
 
def get_response(user_input):
    preprocessed_input = preprocess_text(user_input)
 
    # Check spelling and suggest correction
    corrected_input = str(TextBlob(preprocessed_input).correct())
 
    # Vectorize corrected user input
    X_user = vectorizer.transform([corrected_input])
 
    # Calculate similarity scores
    similarity_scores = cosine_similarity(X_user, X).flatten()
 
    # Find the most similar intent
    max_score_index = similarity_scores.argmax()
 
    # Check if similarity score is above a certain threshold
    if similarity_scores[max_score_index] > 0.5:
        response_options = responses[max_score_index]
        return random.choice(response_options)
    return "I do not understand..."
 
 
if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break
 
        resp = get_response(sentence)
        print(resp)