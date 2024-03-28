import csv
import re
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.decomposition import LatentDirichletAllocation

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

# Funkce pro výpočet podobnosti slov
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

# Funkce pro vyhodnocení úspěšnosti klasifikace
def evaluate_classification(y_test, y_pred):
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=1))  # Přidání parametru zero_division=1

# Tématické modelování s použitím Latent Dirichlet Allocation
def apply_lda(X_train_vec, num_topics=5):
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X_train_vec)
    return lda

# Funkce pro přiřazení tématických prvků ke každé žádosti
def assign_topics(lda_model, X_test_vec):
    return lda_model.transform(X_test_vec)

# Evaluace úspěšnosti klasifikace
def evaluate_application(y_test, y_pred, abbreviations_test, application_file_ano, application_file_ne, reason_file, matching_counts):
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
        
        if y_pred[i] == 0:  # If predicted as negative
            with open(application_file_ne, 'r', encoding='utf-8') as csvfile:
                csvreader = csv.DictReader(csvfile)
                data_ne = [row for row in csvreader]
            for row in data_ne:
                if levenshtein_distance(data_ano[i]['duvod_o_azyl'].lower(), row['duvod_o_azyl'].lower()) <= 2:
                    matching_words.clear()  # Clear matching words if similar reason found in rejected applications
        
        success_rate = (int(bool(matching_words)) + int(matching_counts.get(abbreviation, 0) > 0)) * 50  # 50 for each condition
        success_rates[abbreviation_lower].append(success_rate)
    
    return success_rates

# Funkce pro zpracování a vyhodnocení žádostí s využitím tématického modelování
def process_and_evaluate_applications(train_ano_file, train_ne_file, test_data_file, reason_file, stopwords_file):
    stopwords = load_stopwords(stopwords_file)
    reasons = load_reasons(reason_file)
    
    X_train, y_train, X_test, y_test, abbreviations_train, abbreviations_test = [], [], [], [], [], []
    matching_counts, similar_words = {}, {}
    
    # Načtení a zpracování trénovacích dat
    with open(train_ano_file, 'r', encoding='utf-8') as csvfile:
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
                    X_train.append(' '.join(matching_words))
                    y_train.append(1)  # Positive case
                else:
                    X_train.append(' ')
                    y_train.append(0)  # Negative case
                abbreviations_train.append(abbreviation)
                matching_counts[abbreviation] = len(matching_words)  # Store the count of matching words
                similar_words[abbreviation] = similar  # Store similar words
    
    # Načtení a zpracování testovacích dat
    with open(test_data_file, 'r', encoding='utf-8') as csvfile:
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
                    X_test.append(' '.join(matching_words))
                    y_test.append(1)  # Positive case
                else:
                    X_test.append(' ')
                    y_test.append(0)  # Negative case
                abbreviations_test.append(abbreviation)
    
    # Vektorizace textu
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Aplikace modelu LDA
    lda_model = apply_lda(X_train_vec)

    # Přiřazení témat ke každé žádosti
    topic_assignments = assign_topics(lda_model, X_test_vec)
    print("\nTopic Assignments for Each Request:")
    for i, assignment in enumerate(topic_assignments):
        abbreviation = abbreviations_test[i]
        print(f"Request {abbreviation}: Topic {assignment}")

    # Treshold for classification
    threshold = 0.5

    print("\nClassification based on Threshold:")
    for i, assignment in enumerate(topic_assignments):
        abbreviation = abbreviations_test[i]
        print(sum(assignment))
        print(len(assignment))
        avg_topic = sum(assignment) / len(assignment)
        if avg_topic >= threshold:
            print(f"Request {abbreviation}: Ano")
        else:
            print(f"Request {abbreviation}: Ne")


    # Klasifikace žádostí
    clf = MultinomialNB()
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)

    # Vyhodnocení klasifikace
    evaluate_classification(y_test, y_pred)

    # Vyhodnocení úspěšnosti žádostí
    success_rates = evaluate_application(y_test, y_pred, abbreviations_test, train_ano_file, train_ne_file, reason_file, matching_counts)
    for abbreviation in success_rates:
        average_success_rate = sum(success_rates[abbreviation]) / len(success_rates[abbreviation])
        print(f"{abbreviation.upper()} - {average_success_rate:.2f}%")

    print("\nSimilar Words for Each Request:")
    for abbreviation in abbreviations_test:
        print(f"{abbreviation}: {', '.join(similar_words.get(abbreviation, []))}")

# File paths
train_ano_file = "zadostiSyrieAno.csv"
train_ne_file = "zadostiSyrieNe.csv"
test_data_file = "testovaciData.csv"
reason_file = "syrie.txt"
stopwords_file = "stopwords-cs.json"

# Zpracování a vyhodnocení žádostí
process_and_evaluate_applications(train_ano_file, train_ne_file, test_data_file, reason_file, stopwords_file)
