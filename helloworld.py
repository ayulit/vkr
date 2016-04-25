# coding: utf-8

import pickle
import os

#pickle.load()

data_path1 = os.path.join('OHSUMED','quant_OHSUMED_test_87.txt.pickle')
data_path2 = os.path.basename('OHSUMED')

print data_path1
print data_path2


loadedObj = pickle.load(open(data_path1, 'rb'))

csr = loadedObj[0]
y = loadedObj[1]

print csr