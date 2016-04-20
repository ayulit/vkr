# coding: utf-8

import pickle
import os

#pickle.load()

data_path1 = os.path.join('OHSUMED','quant_OHSUMED_train.arff.pickle')
data_path2 = os.path.basename('OHSUMED')

print data_path1
print data_path2


pickle.load(open(data_path1, 'rb'))