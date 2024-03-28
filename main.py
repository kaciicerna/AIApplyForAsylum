import csv
import re
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

def load_csv_file(file_name):
    data = []
    with open(file_name, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip the header
        for row in csvreader:
            data.append(row)
    return data

def load_reasons(reason_file):
    with open(reason_file, 'r', encoding='utf-8') as file:
        return set(file.read().lower().split(','))

def load_stopwords(stopwords_file):
    with open(stopwords_file, 'r', encoding='utf-8') as file:
        return set(json.load(file))

def levenshtein_distance(s1, s2):
    distances = np.zeros((len(s1) + 1, len(s2) + 1))

    for i in range(len(s1) + 1):
        distances[i][0] = i
    for j in range(len(s2) + 1):
        distances[0][j] = j

    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            distances[i][j] = min(distances[i - 1][j] + 1, distances[i][j - 1] + 1, distances[i - 1][j - 1] + cost)

    return distances[len(s1)][len(s2)]

def evaluate_application(y_test, y_pred, abbreviations_test, application_file_ano, reason_file, matching_counts):
    with open(application_file_ano, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.DictReader(csvfile)
        data_ano = [row for row in csvreader]
    
    reasons = load_reasons(reason_file)
    
    success_rates = {}
    for i, abbreviation in enumerate(abbreviations_test):
        abbreviation_lower = abbreviation.lower()
        if abbreviation_lower not in success_rates:
            success_rates[abbreviation_lower] = []
        
        matching_words = set()
        for word in re.findall(r'\b\w+\b', data_ano[i]['duvod_o_azyl'].lower()):
            min_distance = min(levenshtein_distance(word, reason) for reason in reasons)
            if min_distance <= 2:
                matching_words.add(word)
        
        success_rate = (int(bool(matching_words)) + int(matching_counts.get(abbreviation, 0) > 0)) * 50  # 50 for each condition
        success_rates[abbreviation_lower].append(success_rate)
    
    return success_rates

def evaluate_syria_applications(application_file, reason_file, stopwords_file):
    stopwords = load_stopwords(stopwords_file)
    reasons = load_reasons(reason_file)
    X = []
    y = []
    abbreviations = []
    matching_counts = {}
    similar_words = {}  # Dictionary to store similar words
    
    with open(application_file, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            if row['statni_prislusnost'].lower() in ["sýrie", "syrie"]:
                abbreviation = row['zkratka'].lower()
                application_words = re.findall(r'\b\w+\b', row['duvod_o_azyl'].lower())
                
                matching_words = set()
                similar = set()  # Set to store similar words
                for word in application_words:
                    min_distance = min(levenshtein_distance(word, reason) for reason in reasons)
                    if min_distance <= 2 and word not in stopwords:
                        matching_words.add(word)
                    # Find similar words
                    for reason in reasons:
                        if levenshtein_distance(word, reason) <= 2:
                            similar.add(reason)
                
                if matching_words:
                    X.append(' '.join(matching_words))
                    y.append(1)  # Positive case
                else:
                    X.append(' ')
                    y.append(0)  # Negative case
                abbreviations.append(abbreviation)
                matching_counts[abbreviation] = len(matching_words)  # Store the count of matching words
                similar_words[abbreviation] = similar  # Store similar words
    
    X_train, X_test, y_train, y_test, abbreviations_train, abbreviations_test = train_test_split(X, y, abbreviations, test_size=0.3, random_state=42)
    
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    clf = MultinomialNB()
    clf.fit(X_train_vec, y_train)
    
    y_pred = clf.predict(X_test_vec)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=1))  # Přidání parametru zero_division=1
    print("\nAccuracy for Each Request:")
    success_rates = evaluate_application(y_test, y_pred, abbreviations_test, application_file_ano, reason_file, matching_counts)
    for abbreviation in success_rates:
        average_success_rate = sum(success_rates[abbreviation]) / len(success_rates[abbreviation])
        print(f"{abbreviation.upper()} - {average_success_rate:.2f}%")
        
    print("\nSimilar Words for Each Request:")
    for abbreviation in abbreviations_test:
        print(f"{abbreviation}: {', '.join(similar_words.get(abbreviation, []))}")

def plot_classification_report(y_test, y_pred):
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
    
    labels = ['Negative', 'Positive']
    metrics = [precision, recall, f1_score]
    metric_names = ['Precision', 'Recall', 'F1-score']
   
    for i, metric in enumerate(metrics):
        print(f'{metric_names[i]} by Class:')
        for j, label in enumerate(labels):
            print(f'{label}: {metric[j]}')
        print()

# File paths
application_file_ano = "zadostiSyrieAno.csv"
application_file_ne = "zadostiSyrieNe.csv"
reason_file = "syrie.txt"
stopwords_file = "stopwords-cs.json"

# Using test data
test_data_files = ["testovaciData.csv"]  # Add all test data files here

# Evaluating applications for each test data file
for test_data_file in test_data_files:
    evaluate_syria_applications(test_data_file, reason_file, stopwords_file)