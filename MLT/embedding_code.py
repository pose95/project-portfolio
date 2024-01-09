# -*- coding: utf-8 -*-
"""
Created on Mon May 29 15:13:33 2023

@author: matteo posenato
"""
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

# Definisci il percorso del file di testo di allenamento
input_file = 'yelp-train.txt.ss'

# Leggi il file di testo
with open(input_file, 'r', encoding='utf-8') as f:
    text = f.read()

# Preprocessa il testo
sentences = [simple_preprocess(sentence) for sentence in text.split('\n')]

# Crea il modello Word2Vec
model = Word2Vec(sentences, vector_size=300, window=5, min_count=1, workers=4)

# Salva il modello Word2Vec su file
output_file = 'embedding_yelp_300.txt'
model.wv.save_word2vec_format(output_file, binary=False)

