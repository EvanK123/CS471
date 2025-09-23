import csv
import re
import math

data = []
with open('SpamDetection.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    for row in reader:
        data.append(row)

# Split into training (first 20 samples) and testing (last 10 samples)
train_data = data[:20]
test_data = data[20:]

def preprocess_text(text):
    # Convert to lowercase and remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    # Tokenize into words
    words = text.split()
    return words

# Preprocess all messages
train_tokens = []
for row in train_data:
    tokens = preprocess_text(row[1])
    train_tokens.append((row[0], tokens))

test_tokens = []
for row in test_data:
    tokens = preprocess_text(row[1])
    test_tokens.append((row[0], tokens))

# Count spam and ham in training data
spam_count = 0
ham_count = 0
for target, _ in train_tokens:
    if target == 'spam':
        spam_count += 1
    else:
        ham_count += 1
total_count = len(train_tokens)

# Calculate prior probabilities
p_spam = spam_count / total_count
p_ham = ham_count / total_count

print(f"P(spam) = {p_spam:.4f}")
print(f"P(ham) = {p_ham:.4f}")

# Build vocabulary from training data
vocabulary = set()
spam_words = []
ham_words = []

for target, tokens in train_tokens:
    if target == 'spam':
        spam_words.extend(tokens)
    else:
        ham_words.extend(tokens)
    vocabulary.update(tokens)

vocabulary_size = len(vocabulary)

# Count word frequencies with Laplace smoothing (add-1)
spam_word_count = len(spam_words)
ham_word_count = len(ham_words)

# Create dictionaries for word probabilities
spam_word_probs = {}
ham_word_probs = {}

for word in vocabulary:
    # Count occurrences in spam and ham with Laplace smoothing
    spam_count_word = spam_words.count(word) + 1
    ham_count_word = ham_words.count(word) + 1
    
    # Calculate probabilities
    spam_word_probs[word] = spam_count_word / (spam_word_count + vocabulary_size)
    ham_word_probs[word] = ham_count_word / (ham_word_count + vocabulary_size)

def classify_message(tokens):
    # Initialize probabilities with priors
    spam_prob = math.log(p_spam)
    ham_prob = math.log(p_ham)
    
    # Add log probabilities of each word
    for word in tokens:
        if word in vocabulary:
            spam_prob += math.log(spam_word_probs[word])
            ham_prob += math.log(ham_word_probs[word])
        # If word not in vocabulary, we skip it (doesn't affect probabilities)
    
    # Convert back from log space
    spam_prob = math.exp(spam_prob)
    ham_prob = math.exp(ham_prob)
    
    # Normalize probabilities
    total_prob = spam_prob + ham_prob
    spam_prob_normalized = spam_prob / total_prob
    ham_prob_normalized = ham_prob / total_prob
    
    # Determine classification
    if spam_prob_normalized > ham_prob_normalized:
        classification = 'spam'
    else:
        classification = 'ham'
    
    return spam_prob_normalized, ham_prob_normalized, classification

# Test the classifier on test data
correct_predictions = 0
total_predictions = len(test_tokens)

print("\nTest Results:")
print("=" * 80)
for i, (actual, tokens) in enumerate(test_tokens):
    spam_prob, ham_prob, predicted = classify_message(tokens)
    
    # Check if prediction is correct
    is_correct = "✓" if predicted == actual else "✗"
    if predicted == actual:
        correct_predictions += 1
    
    print(f"Message {i+1}: {test_data[i][1][:50]}...")
    print(f"  Actual: {actual}, Predicted: {predicted} {is_correct}")
    print(f"  P(spam|message): {spam_prob:.6f}, P(ham|message): {ham_prob:.6f}")
    print("-" * 80)

# Calculate accuracy
accuracy = correct_predictions / total_predictions
print(f"\nAccuracy: {correct_predictions}/{total_predictions} = {accuracy:.2%}")