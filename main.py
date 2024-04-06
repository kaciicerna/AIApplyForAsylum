import csv
import re
import json
import numpy as np
from stemmer import cz_stem
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt

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

# Sloupcový graf průměrné úspěšnosti pro každou zkratku
def plot_similar_words(similar_words):
    # Sečíst počet výskytů každého slova
    word_counts = {}
    for words in similar_words.values():
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
    
    # Seřadit slova podle četnosti
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    top_words = [word[0] for word in sorted_words[:10]]  # Vybrat prvních 10 nejčastějších slov
    
    # Vytvořit sloupcový graf
    plt.figure(figsize=(10, 6))
    plt.bar(top_words, [word_counts[word] for word in top_words])
    plt.title('Četnost konkrétních slov (similar words) ve všech žádostech')
    plt.xlabel('Slovo')
    plt.ylabel('Počet výskytů')
    plt.xticks(rotation=45)
    plt.show()

# Funkce pro vyhodnocení úspěšnosti klasifikace
def evaluate_classification(y_test, y_pred):
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=1))  # Přidání parametru zero_division=1

def evaluate_application(y_test, y_pred, abbreviations_test, train_ano_file, train_ne_file, reason_file, matching_counts):
    with open(train_ano_file, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.DictReader(csvfile)
        data_ano = [row for row in csvreader]
    
    with open(train_ne_file, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.DictReader(csvfile)
        data_ne = [row for row in csvreader]
    
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
        similar_to_negative_data = any(levenshtein_distance(data_ano[i]['duvod_o_azyl'].lower(), row['duvod_o_azyl'].lower()) <= 3 for row in data_ne)
        
        # Vyhodnocení úspěšnosti žádosti na základě nových kritérií
        if (y_pred[i] == 0 or 
            data_ano[i]['podepsane_prohlaseni'].lower() == "ne" or 
            data_ano[i]['duvod_o_azyl'] == "" or 
            data_ano[i]['doklady_dokumenty'] == ""):
            success_rate = 0  # Klasifikátor předpověděl negativní výsledek nebo podepsané prohlášení s "ne" -> 0%
        elif similar_to_training:
            success_rate = 70  # Data jsou shodná nebo podobná pozitivním trénovacím datům -> 70%
        elif similar_to_negative_data:
            success_rate = 0 # Data jsou shodná nebo podobná negativním trénovacím datům -> 0%
        else:
            success_rate = 0  # Defaultní hodnota úspěšnosti
        
        # Přičtení dalších procent za shodná slova
        success_rate += matching_words_count * 5
        
        # Omezení maximální hodnoty na 100%
        success_rate = min(success_rate, 100)
        
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
    print(y_pred)
    
    # Vyhodnocení klasifikace
    # evaluate_classification(y_test, y_pred)

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
        
     # Volání funkcí pro vykreslení grafů
    plot_similar_words(similar_words)
    
# File paths
train_ano_file = "zadostiSyrieAno.csv"
train_ne_file = "zadostiSyrieNe.csv"
test_data_file = "testovaciData2.csv"
reason_file = "syrie.txt"
stopwords_file = "stopwords-cs.json"

# Zpracování a vyhodnocení žádostí
process_and_evaluate_applications(train_ano_file, train_ne_file, test_data_file, reason_file, stopwords_file)
