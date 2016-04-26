# coding: utf-8
"""Загружает pickle c разреженными матрицами

   Скрипт рассчитан на работу с корпусами OHSUMED и RCV

   В настоящий момент не используется, т.к. данные напрямую преобразуются из arff в UCI

"""

import pickle
import os

corpus_data_folder = os.path.join('rawdata','OHSUMED')
pickle_data_folder = os.path.join(corpus_data_folder,'pickles')
pickle_data_file = os.path.join(pickle_data_folder,'quant_OHSUMED_test_87.txt.pickle')
uci_data_folder = os.path.join('train','ohsumed')

# создать папку для UCI данных, если её нет
if not os.path.exists(uci_data_folder):
    os.makedirs(uci_data_folder)

file = open(pickle_data_file, 'rb')
# [csr, y, y_names]
# csr - разреженая матрица TFIDF (документ-термин)
# y -   разреженная бинарная матрица с индексами классов (документ-класс)
# y_names - имена классов
pickle_data_object = pickle.load(file)
file.close()



csr = pickle_data_object[0]
y = pickle_data_object[1]
y_names = pickle_data_object[2]

# не работает
# print csr.csr_matrix.getnnz([0])


