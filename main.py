import csv
import re
import json
import numpy as np

def nacti_csv_soubor(nazev_souboru):
    data = []
    with open(nazev_souboru, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.DictReader(csvfile)
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
    # Create a matrix to store distances
    distances = np.zeros((len(s1) + 1, len(s2) + 1))

    # Initialize the first row and column
    for i in range(len(s1) + 1):
        distances[i][0] = i
    for j in range(len(s2) + 1):
        distances[0][j] = j

    # Calculate distances
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            distances[i][j] = min(distances[i - 1][j] + 1, distances[i][j - 1] + 1, distances[i - 1][j - 1] + cost)

    return distances[len(s1)][len(s2)]

def zjisti_duvody_syrie(soubor_zadosti, soubor_duvodu, soubor_stopwords):
    stopwords = nacti_stopwords(soubor_stopwords)
    slova_duvody = nacti_duvody(soubor_duvodu)
    
    with open(soubor_zadosti, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            if row['statni_prislusnost'].lower() in ["sýrie", "syrie"]:
                zkratka = row['zkratka'].lower()
                slova_zadosti = re.findall(r'\b\w+\b', row['duvod_o_azyl'].lower())
                
                # Find similar words from Sýrie.txt in the words from the application
                shodna_slova = set()
                for slovo_zadosti in slova_zadosti:
                    min_distance = min(levenshtein_distance(slovo_zadosti, slovo_duvodu) for slovo_duvodu in slova_duvody)
                    if min_distance <= 2 and slovo_zadosti not in stopwords:
                        shodna_slova.add(slovo_zadosti)
                
                if shodna_slova:
                    print('"{}" = {}'.format(zkratka.upper(), list(shodna_slova)))
                else:
                    print('"{}" = žádná slova'.format(zkratka.upper()))

soubor_zadosti = "zadostiSyrieAno.csv"
soubor_duvodu = "syrie.txt"
soubor_stopwords = "stopwords-cs.json"

zjisti_duvody_syrie(soubor_zadosti, soubor_duvodu, soubor_stopwords)
