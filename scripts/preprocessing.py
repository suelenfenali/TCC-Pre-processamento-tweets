# coding: utf-8

import pandas as pd
import numpy as np
import nltk
import csv
import spell
import editdistance

import sys  

reload(sys)  
sys.setdefaultencoding('utf8')

from nltk import *
from pyjarowinkler import distance
from metaphone import doublemetaphone

# função de limpeza do tweet - primeiro passo

def clean_text(text):
    t = [ ]
    for w in text.split(' '):
        if not (w.startswith('http') or w.startswith('@') or w.startswith('#')):
            t.append(w)
    
    return ' '.join(t)

#############################
words = []
stops = []
sentences = []
normalised_sentences = []
iv_words = []
oov_words = []
non_words = []
num_tweets_normalised = 0
normalised_words = {}

# abre arquivo de tweets
with open('tweetsEnglish.csv') as csvfile:
	main_file = csvfile.readlines()


# limpa stopwords e tokeniza
stopwords = nltk.corpus.stopwords.words('english')
tokenizer = nltk.tokenize.RegexpTokenizer('[a-zA-Z]\w+')


for row in main_file:
	sentence = clean_text(row)
	sentence = tokenizer.tokenize(sentence.decode('utf-8'))
	for token in sentence:
		if token not in stopwords:
			words.append(token)
		else:
			stops.append(token)
	sentences.append(sentence)
	
print("Finalizou tokenização")

##############################

# verifica IV e OOV

for word in words:
	if word[0].isupper():
		non_words.append(word)
	elif not nltk.corpus.wordnet.synsets(word):
		oov_words.append(word)
	else:
		iv_words.append(word)

print("Finalizou separação IV, OVV, NON_V")
##############################

# Normaliza

# abre arquivo de palavras lexicamente corretas
with open('words.txt') as words_dictionary:
	word_dict = words_dictionary.readlines()

for oov in oov_words:
	first_set = []
	phonetic_matches = []
	
	for entry in word_dict:
		distance_lv = editdistance.eval(entry, oov)
# 		distance_jw = distance.get_jaro_distance(entry, oov, winkler=True, scaling=0.1)
		if distance_lv <= 2: #0.91:
			first_set.append(entry)
# 	print("Finalizou Levensthein")
	
	for similar in first_set:
		if (doublemetaphone(oov) == doublemetaphone(similar)):
			phonetic_matches.append(similar)
# 	print("Finalizou metaphone")

	peter_suggestion = spell.correction(oov)
# 	print("Finalizou Peter Norvig")
	
	found = 'false'
	if len(phonetic_matches) == 0:
		normalised_words[oov] = peter_suggestion
	elif len(phonetic_matches) == 1:
		normalised_words[oov] = phonetic_matches[0]
	elif len(phonetic_matches) > 1:
		for match in phonetic_matches:
			if match == peter_suggestion:
				normalised_words[oov] = match
				found = 'true'
		if found == 'false':
			dist_lev = editdistance.eval(phonetic_matches[0], oov)
			minor_dist_lev = phonetic_matches[0]
			for match in phonetic_matches:
				current_dist_lev = editdistance.eval(match, oov)
				if current_dist_lev <= dist_lev:
					minor_dist_lev = match
					dist_lev = current_dist_lev
			normalised_words[oov] = minor_dist_lev
	
	print("Next OOV")

print("Finalizou normalização")
############################################################

# Altera tweets
print("Inicializou alteração dos tokens nos tweets")

write_results = open('normalised_output.txt', 'w')

for row in main_file:
	sentence_non_normalised = clean_text(row)
	sentence_normalised = sentence_non_normalised
	sentence_non_normalised = tokenizer.tokenize(sentence_non_normalised.decode('utf-8'))
	new_sentence = sentence_normalised
	for token in sentence_non_normalised:
		if token in oov_words:
			new_sentence = new_sentence.replace(token, normalised_words[token])
	normalised_sentences.append(new_sentence)
	write_results.write(new_sentence + "\n")
	if sentence_normalised != sentence_normalised:
		num_tweets_normalised = num_tweets_normalised + 1

############################################################

# Cria estatísticas
print("Inicializou criação de estatísticas")


fd_oov = FreqDist(oov_words)
fd_oov.most_common()
fd_oov.plot(30, cumulative=False)

fd_iv = FreqDist(iv_words)
fd_iv.most_common()
fd_iv.plot(30, cumulative=False)

print('Fim!')
