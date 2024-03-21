import csv
import re
import json

def nacti_csv_soubor(nazev_souboru):
    data = []
    with open(nazev_souboru, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            data.append(row)
    return data

def nacti_duvody(soubor_duvodu):
    with open(soubor_duvodu, 'r', encoding='utf-8') as file:
        return set(file.read().lower().split())

def nacti_stopwords(soubor_stopwords):
    with open(soubor_stopwords, 'r', encoding='utf-8') as file:
        return set(json.load(file))

def zjisti_duvody_syrie(soubor_zadosti, soubor_duvodu, soubor_stopwords):
    stopwords = nacti_stopwords(soubor_stopwords)
    slova_duvody = nacti_duvody(soubor_duvodu)
    
    with open(soubor_zadosti, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            if row['statni_prislusnost'].lower() in ["sýrie", "syrie"]:
                zkratka = row['zkratka'].lower()
                slova_zadosti = re.findall(r'\b\w+\b', row['duvod_o_azyl'].lower())
                
                # Hledáme shodná slova nebo kořeny slov z Sýrie.txt ve slovech žádosti
                shodna_slova = set()
                for slovo_zadosti in slova_zadosti:
                    for slovo_duvodu in slova_duvody:
                        if slovo_zadosti.startswith(slovo_duvodu) and slovo_zadosti.lower() not in stopwords:
                            shodna_slova.add(slovo_zadosti)
                            break
                
                if shodna_slova:
                    print('"{}" = {}'.format(zkratka.upper(), list(shodna_slova)))
                else:
                    print('"{}" = žádná slova'.format(zkratka.upper()))

soubor_zadosti = "zadostiSyrieAno.csv"
soubor_duvodu = "syrie.txt"
soubor_stopwords = "stopwords-cs.json"

zjisti_duvody_syrie(soubor_zadosti, soubor_duvodu, soubor_stopwords)
