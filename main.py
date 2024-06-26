import csv
import re
import json
import numpy as np
from stemmer import cz_stem
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
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
    word_counts = {}
    for words in similar_words.values():
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
    # Seřadit slova podle četnosti
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    top_words = [word[0] for word in sorted_words[:10]]
    
    plt.figure(figsize=(10, 6))
    plt.bar(top_words, [word_counts[word] for word in top_words])
    plt.title('Četnost konkrétních slov (similar words) ve všech žádostech')
    plt.xlabel('Slovo')
    plt.ylabel('Počet výskytů')
    plt.xticks(rotation=45)

    for i, v in enumerate(top_words):
        plt.text(i, word_counts[v] + 0.2, str(word_counts[v]), ha='center', va='bottom')

    plt.show()
    
def plot_success_failure_by_country(success_rates):
    countries = list(success_rates.keys())
    success_rates_values = [sum(success_rates[country]) / len(success_rates[country]) for country in countries]
    failure_rates_values = [100 - rate for rate in success_rates_values]

    plt.figure(figsize=(10, 6))
    plt.bar(countries, success_rates_values, label='Úspěch', color='g', alpha=0.6)
    plt.bar(countries, failure_rates_values, bottom=success_rates_values, label='Neúspěch', color='r', alpha=0.6)
    plt.xlabel('Země')
    plt.ylabel('Procento')
    plt.title('Míra úspěšnosti konkrétních žádostí')
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

# Funkce pro vyhodnocení úspěšnosti klasifikace
def evaluate_classification(y_test, y_predicted):
    print("Classification Report:")
    print(classification_report(y_test, y_predicted, zero_division=1))

def evaluate_application(y_test, y_predicted, abbreviations_test, train_yes_file, train_no_file, reasons, matching_counts):
    with open(train_yes_file, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.DictReader(csvfile)
        data_yes = [row for row in csvreader]
    
    with open(train_no_file, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.DictReader(csvfile)
        data_no = [row for row in csvreader]
    
    success_rates = {}
    for i, abbreviation in enumerate(abbreviations_test):
        abbreviation_lower = abbreviation.lower()
        if abbreviation_lower not in success_rates:
            success_rates[abbreviation_lower] = []
        
        # Proces kontroly
        if data_yes[i]['doklady_dokumenty'] == "" or data_yes[i]['podepsane_prohlaseni'].lower() == "ne":
            success_rate = 0
        else:
            matching_words = set()
            for word in re.findall(r'\b\w+\b', data_yes[i]['duvod_o_azyl'].lower()):
                stemmed_word = cz_stem(word) # Stemmatizace slova
                min_distance = min(levenshtein_distance(stemmed_word, reason) for reason in reasons)
                if min_distance <= 3:
                    matching_words.add(word)
                    
            # Zjištění, zda je text podobný trénovacím datům
            similar_to_training = bool(matching_counts.get(abbreviation, 0) > 0)
            # Zjištění skutečného počtu shodných slov
            matching_words_count = len(matching_words)
            
            # Zjištění, zda je text podobný negativním trénovacím datům
            similar_to_negative_data = any(levenshtein_distance(data_yes[i]['duvod_o_azyl'].lower(), row['duvod_o_azyl'].lower()) <= 3 for row in data_no)
            
            # Vyhodnocení úspěšnosti žádosti na základě nových kritérií
            if y_predicted[i] == 0 or similar_to_negative_data:
                success_rate = 0  # Klasifikátor předpověděl negativní výsledek nebo data jsou podobná negativním trénovacím datům -> 0%
            elif similar_to_training:
                success_rate = 70  # Data jsou shodná nebo podobná pozitivním trénovacím datům -> 70%
            else:
                success_rate = 0  # Defaultní hodnota úspěšnosti
            
            # Přičtení dalších procent za shodná slova
            success_rate += matching_words_count * 5
            # Omezení maximální hodnoty na 99%
            success_rate = min(success_rate, 99)
        
        # Přidání vypočtené úspěšnosti do seznamu
        success_rates[abbreviation_lower].append(success_rate)
        
        # Výpis počtu shodných slov vedle úspěšnosti žádosti
        if matching_words_count > 3 and success_rate < 50:
            print(f"{abbreviation.upper()} - {success_rate:.2f}%, shodná slova {matching_words_count}/{(20)} - doporučeno ke kontrole")
        else:
            # Výpis počtu shodných slov vedle úspěšnosti žádosti
            print(f"{abbreviation.upper()} - {success_rate:.2f}%, shodná slova {matching_words_count}/{(20)}")

    return success_rates

def process_and_evaluate_applications(train_yes_file, train_no_file, test_data_file, reason_file, stopwords_file, country):
    stopwords = load_stopwords(stopwords_file)
    reasons = load_reasons(reason_file)
    
    X_train, y_train, X_test, y_test, abbreviations_train, abbreviations_test = [], [], [], [], [], []
    matching_counts, similar_words = {}, {}

    # Zpracování trénovacích dat pro danou zemi
    with open(train_yes_file, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            if row['statni_prislusnost'].lower() == country.lower():
                abbreviation = row['zkratka'].lower()
                text = row['duvod_o_azyl'].lower()
                filtered_text = ' '.join(word for word in re.findall(r'\b\w+\b', text) if word not in stopwords)
                X_train.append(filtered_text)
                y_train.append(1)
                abbreviations_train.append(abbreviation)
                matching_counts[abbreviation] = len(re.findall(r'\b\w+\b', text))
    
    with open(train_no_file, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            if row['statni_prislusnost'].lower() == country.lower():
                abbreviation = row['zkratka'].lower()
                text = row['duvod_o_azyl'].lower()
                filtered_text = ' '.join(word for word in re.findall(r'\b\w+\b', text) if word not in stopwords)
                X_train.append(filtered_text)
                y_train.append(0)
                abbreviations_train.append(abbreviation)
                
    # Zpracování testovacích dat pro danou zemi
    with open(test_data_file, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            if row['statni_prislusnost'].lower() == country.lower():
                abbreviation = row['zkratka'].lower()
                text = row['duvod_o_azyl'].lower()
                filtered_text = ' '.join(word for word in re.findall(r'\b\w+\b', text) if word not in stopwords)
                X_test.append(filtered_text)
                if abbreviation in matching_counts:
                    y_test.append(1)
                else:
                    y_test.append(0)
                abbreviations_test.append(abbreviation)
                similar_words[abbreviation] = set()

    # Vektorizace textu
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Klasifikace žádostí
    clf = MultinomialNB()
    clf.fit(X_train_vec, y_train)
    y_predicted = clf.predict(X_test_vec)
    print(y_predicted)
    
    # Vyhodnocení klasifikace
    evaluate_classification(y_test, y_predicted)

    # Vyhodnocení úspěšnosti žádostí
    print(f"\n{'=' * 20} {country.upper()} {'=' * 20}")
    success_rates = evaluate_application(y_test, y_predicted, abbreviations_test, train_yes_file, train_no_file, reasons, matching_counts)
    for abbreviation in success_rates:
        average_success_rate = sum(success_rates[abbreviation]) / len(success_rates[abbreviation])
        print(f"{abbreviation.upper()} - {average_success_rate:.2f}%")

    # Vyhledání podobných slov
    for abbreviation, request in zip(abbreviations_test, X_test):
        for word in re.findall(r'\b\w+\b', request):
            stemmed_word = cz_stem(word)
            for reason in reasons:
                if levenshtein_distance(stemmed_word, reason) <= 3:
                    similar_words[abbreviation].add(word)

    # Výstup podobných slov pro každou žádost
    print("\nSimilar Words for Each Request:")
    for abbreviation in abbreviations_test:
        print(f"{abbreviation}: {', '.join(similar_words.get(abbreviation, []))}")

    # Volání funkcí pro vykreslení grafů
    plot_similar_words(similar_words)
    
    plot_success_failure_by_country(success_rates)

# File paths
train_syrie_yes_file = "zadostSyrieAno.csv"
train_syrie_no_file = "zadostSyrieNe.csv"
train_irak_yes_file = "zadostIrakAno.csv"
train_tunis_yes_file = "zadostTunisAno.csv"
train_afghanistan_yes_file = "zadostAfghanistanAno.csv"
train_irak_no_file = "zadostIrakNe.csv"
train_tunis_no_file = "zadostTunisNe.csv"
train_afghanistan_no_file = "zadostAfghanistanNe.csv"
test_data_file = "testovaciDataF.csv"
reason_syrie_file = "syrie.txt"
reason_irak_file = "irak.txt"
reason_tunis_file = "tunis.txt"
reason_afghanistan_file = "afganistan.txt"
stopwords_file = "stopwords-cs.json"

### Zpracování a vyhodnocení žádostí
# Volání funkce pro Syrii
process_and_evaluate_applications(train_syrie_yes_file, train_syrie_no_file, test_data_file, reason_syrie_file, stopwords_file, "sýrie")
# Volání funkce pro Irák
process_and_evaluate_applications(train_irak_yes_file, train_irak_no_file, test_data_file, reason_irak_file, stopwords_file, "irák")
# Volání funkce pro Tunisko
process_and_evaluate_applications(train_tunis_yes_file, train_tunis_no_file, test_data_file, reason_tunis_file, stopwords_file, "tunis")
# Volání funkce pro Afghánistán
process_and_evaluate_applications(train_afghanistan_yes_file, train_afghanistan_no_file, test_data_file, reason_afghanistan_file, stopwords_file, "afghánistán")