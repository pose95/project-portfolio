# -*- coding: utf-8 -*-
"""
Created on Mon May 15 16:58:54 2023

@author: matteo posenato
"""
import json
import numpy as np
import scipy.sparse as sp
import joblib

def read_text(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def write_file(file_path, content):
    with open(file_path, 'w') as file:
        file.write(content)

def copy_file(source_path, destination_path):
    content = read_file(source_path)
    write_file(destination_path, content)

def read_json(nome_file):
    with open(nome_file, 'r') as file:
        dati_json = json.load(file)
    return dati_json

def save_sparse(array, filename):
    joblib.dump(array, filename)

def write_to_json(filename, data):
    with open(data, 'w') as file:
        json.dump(filename, file)

def write_list_to_text(data, filename):
    with open(filename, 'w') as file:
        for item in data:
            file.write(str(item) + '\n')

def log(log_file, message):
    with open(log_file, "a") as file:
        file.write(message + "\n")
        
def load_sparse(filename):
    sparse_matrix = joblib.load(filename)
    return sparse_matrix