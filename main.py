import csv
import re
import json
import numpy as np
from stemmer import cz_stem
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

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

# Funkce pro vyhodnocení úspěšnosti klasifikace
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
            stemmed_word = cz_stem(word)  # Stemmatizace slova
            min_distance = min(levenshtein_distance(stemmed_word, reason) for reason in reasons)
            if min_distance <= 3:
                matching_words.add(word)
        
        # Zjištění, zda je text podobný trénovacím datům
        similar_to_training = bool(matching_counts.get(abbreviation, 0) > 0)
        
        # Zjištění skutečného počtu shodných slov
        matching_words_count = len(matching_words)
        
        # Zjištění, zda je text podobný negativním trénovacím datům
        similar_to_negative_data = False
        if y_pred[i] == 0:  # If predicted as negative
            with open(application_file_ne, 'r', encoding='utf-8') as csvfile:
                csvreader = csv.DictReader(csvfile)
                data_ne = [row for row in csvreader]
            for row in data_ne:
                if levenshtein_distance(data_ano[i]['duvod_o_azyl'].lower(), row['duvod_o_azyl'].lower()) <= 2:
                    similar_to_negative_data = True
                    break
        
        # Vyhodnocení úspěšnosti žádosti na základě více faktorů
        if data_ano[i]['duvod_o_azyl'].strip() == "":
            success_rate = 0  # Pokud je důvod o azyl prázdný -> 0%
            success_rates[abbreviation_lower].append(success_rate)
            continue
        
        if similar_to_training:
            success_rate = 100  # Data jsou shodná nebo podobná trénovacím datům -> 100%
        elif matching_words_count > 0:
            if matching_words_count >= 3:
                success_rate = 60  # Data obsahují 3 nebo více shodných slov -> 60%
            else:
                success_rate = 30  # Data obsahují slova ze souboru Sýrie.txt -> 30%
        elif similar_to_negative_data:
            success_rate = 0  # Data jsou podobná negativním trénovacím datům -> 0%
        else:
            success_rate = 0  # Defaultní hodnota úspěšnosti (pro situace, které neodpovídají žádnému z předchozích kritérií)
        
        # Přidání vypočtené úspěšnosti do seznamu
        success_rates[abbreviation_lower].append(success_rate)
    
    return success_rates

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
                X_train.append(row['duvod_o_azyl'].lower())
                y_train.append(1)  # Positive case
                abbreviations_train.append(abbreviation)
                matching_counts[abbreviation] = len(re.findall(r'\b\w+\b', row['duvod_o_azyl'].lower()))  # Store the count of matching words
    
    with open(train_ne_file, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            if row['statni_prislusnost'].lower() in ["sýrie", "syrie"]:
                abbreviation = row['zkratka'].lower()
                X_train.append(row['duvod_o_azyl'].lower())
                y_train.append(0)  # Negative case
                abbreviations_train.append(abbreviation)
    
    # Načtení a zpracování testovacích dat
    with open(test_data_file, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            if row['statni_prislusnost'].lower() in ["sýrie", "syrie"]:
                abbreviation = row['zkratka'].lower()
                X_test.append(row['duvod_o_azyl'].lower())
                # Zde se ukládají binární hodnoty
                if abbreviation in matching_counts:
                    y_test.append(1)  # Positive case
                else:
                    y_test.append(0)  # Negative case
                abbreviations_test.append(abbreviation)
                similar_words[abbreviation] = set()  # inicializace prázdného setu pro každou zkratku

    # Vektorizace textu
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

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

    # Vyhledání podobných slov
    for abbreviation, request in zip(abbreviations_test, X_test):
        for word in re.findall(r'\b\w+\b', request):
            stemmed_word = cz_stem(word)  # Stemmatizace slova
            for reason in reasons:
                if levenshtein_distance(stemmed_word, reason) <= 3:
                    similar_words[abbreviation].add(word)

    # Výstup podobných slov pro každou žádost
    print("\nSimilar Words for Each Request:")
    for abbreviation in abbreviations_test:
        print(f"{abbreviation}: {', '.join(similar_words.get(abbreviation, []))}")

# File paths
train_ano_file = "zadostiSyrieAno.csv"
train_ne_file = "zadostiSyrieNe.csv"
test_data_file = "testovaciData2.csv"
reason_file = "syrie.txt"
stopwords_file = "stopwords-cs.json"

# Zpracování a vyhodnocení žádostí
process_and_evaluate_applications(train_ano_file, train_ne_file, test_data_file, reason_file, stopwords_file)
