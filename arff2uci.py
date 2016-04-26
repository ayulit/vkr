# coding: utf-8
"""Запуск парсинга ARFF

   Скрипт настраивает пути к данным и запускает методы класса по парсингу ARFF

"""

import os
from parse_arff import Parse_ARFF

pa = Parse_ARFF()

corpus_data_folder = os.path.join('rawdata','OHSUMED')
arff_data_folder = os.path.join(corpus_data_folder,'arff')
arff_data_file = os.path.join(arff_data_folder,'quant_OHSUMED_test_87.txt')

#pa.convert_arff(arff_data_folder)
#data_object = pa.make_csr(pa.read_arff(arff_data_file))
#print data_object
#csr = data_object[0]
#y = data_object[1]
#y_names = data_object[2]
#print csr


if pa.make_uci(pa.read_arff(arff_data_file)):
    print "Parsing OK."
