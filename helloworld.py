# coding: utf-8

import pickle
import os

#pickle.load()

data_folder = os.path.join('rawdata','OHSUMED')
data_file = os.path.join(data_folder,'quant_OHSUMED_test_87.txt.pickle')


loadedObj = pickle.load(open(data_file, 'rb'))

csr = loadedObj[0]
y = loadedObj[1]
y_names = loadedObj[2]

print y_names