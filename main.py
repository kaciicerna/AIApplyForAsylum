import csv
import re
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

def nacti_csv_soubor(nazev_souboru):
    data = []
    with open(nazev_souboru, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Přeskočení prvního řádku (hlavička)
        for row in csvreader:
            data.append(row)
    return data

def nacti_duvody(soubor_duvodu):
    with open(soubor_duvodu, 'r', encoding='utf-8') as file:
        return set(file.read().lower().split(','))

def nacti_stopwords(soubor_stopwords):
    with open(soubor_stopwords, 'r', encoding='utf-8') as file:
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

def vypocti_uspesnost_zadosti(y_test, y_pred, zkratky_test):
    uspesnosti = {}
    for i, zkratka in enumerate(zkratky_test):
        zkratka_lower = zkratka.lower()
        if zkratka_lower not in uspesnosti:
            uspesnosti[zkratka_lower] = []
        uspesnost = int(y_pred[i] == y_test[i]) * 100
        uspesnosti[zkratka_lower].append(uspesnost)
        print(f"{zkratka_lower} - {uspesnost:.2f}%")  # Přidáno pro výpis úspěšnosti každé žádosti
    
    return uspesnosti

def zjisti_duvody_syrie(soubor_zadosti, soubor_duvodu, soubor_stopwords):
    stopwords = nacti_stopwords(soubor_stopwords)
    slova_duvody = nacti_duvody(soubor_duvodu)
    X = []
    y = []
    zkratky = []
    
    with open(soubor_zadosti, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.DictReader(csvfile)
        
        # Přeskočení hlavičky
        next(csvreader)
        
        for row in csvreader:
            if row['statni_prislusnost'].lower() in ["sýrie", "syrie"]:
                zkratka = row['zkratka'].lower()
                slova_zadosti = re.findall(r'\b\w+\b', row['duvod_o_azyl'].lower())
                
                shodna_slova = set()
                for slovo_zadosti in slova_zadosti:
                    min_distance = min(levenshtein_distance(slovo_zadosti, slovo_duvodu) for slovo_duvodu in slova_duvody)
                    if min_distance <= 2 and slovo_zadosti not in stopwords:
                        shodna_slova.add(slovo_zadosti)
                
                if shodna_slova:
                    X.append(' '.join(shodna_slova))
                    y.append(1)  # Positive case
                else:
                    X.append(' ')
                    y.append(0)  # Negative case
                zkratky.append(zkratka)
    
    X_train, X_test, y_train, y_test, zkratky_train, zkratky_test = train_test_split(X, y, zkratky, test_size=0.3, random_state=42)
    
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    clf = MultinomialNB()
    clf.fit(X_train_vec, y_train)
    
    y_pred = clf.predict(X_test_vec)
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division='warn'))
    print("\nAccuracy for Each Request:")
    for i, zkratka in enumerate(zkratky_test):
        print(f"{zkratka} - {int(y_pred[i] == y_test[i]) * 100}%")

    # Výpočet úspěšnosti každé žádosti z testovacích dat
    uspesnost_zadosti = vypocti_uspesnost_zadosti(y_test, y_pred, zkratky_test)
    
    # Výpis úspěšnosti pro každou žádost v testovacích datech
    print("\nAccuracy for Each Request:")
    for zkratka in zkratky_test:
        zkratka_lower = zkratka.lower()
        zkratka_upper = zkratka.upper()
        prumer_uspesnosti = (sum(uspesnost_zadosti[zkratka_lower]) + sum(uspesnost_zadosti[zkratka_lower])) / len(uspesnost_zadosti[zkratka_lower])
    print(f"{zkratka_upper} - {prumer_uspesnosti:.2f}%")

    # Vizualizace výsledků
    plot_classification_report(y_test, y_pred)

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

soubor_zadosti_ano = "zadostiSyrieAno.csv"
soubor_zadosti_ne = "zadostiSyrieNe.csv"
soubor_duvodu = "syrie.txt"
soubor_stopwords = "stopwords-cs.json"

# Použití testovacích dat
testovaci_data = "testovaciDataa.csv"
zjisti_duvody_syrie(testovaci_data, soubor_duvodu, soubor_stopwords)