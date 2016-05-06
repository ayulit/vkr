# -*- coding: utf-8 -*-
"""Эксперимент по обработке данных UCI с мультимодальностью для BigARTM v 0.7.5

   Для начала коллекция представлена данными quant_OHSUMED_test_87.txt

   На основе примера
   "Построение тематической модели классификации коллекции EUR-lex"
   https://github.com/bigartm/bigartm-book/blob/master/applications/eurlex/Main_RU.ipynb

   Пока как мне надо настроить не смог

"""

import os
import sys
import glob
import pickle
import numpy
import sklearn.metrics

import artm

labels_class = '@class'
tokens_class = '@default_class'

collection_name = 'kos'
_dictionary_name = 'dictionary'
dictionary_filename = _dictionary_name + '.dict'
target_folder = os.path.join('target', collection_name)
train_data_folder = os.path.join('train', collection_name)  # папка с обучающей выборкой
_dictionary_path = os.path.join(target_folder, dictionary_filename )

num_topics = 3
num_collection_passes = 5

num_document_passes   = [16] * num_collection_passes
labels_class_weight   = [1.0, 1.0, 0.9, 0.9, 0.9, 0.8, 0.8, 0.8, 0.7, 0.7]
tokens_class_weight   = [1] * num_collection_passes

smooth_theta_tau      = [0.02] * num_collection_passes
smooth_phi_tau        = [0.01] * num_collection_passes

smooth_psi_tau        = [0.01] * num_collection_passes
label_psi_tau         = [0.0] * num_collection_passes

count_scores_iters = [num_collection_passes - 1]




batch_vectorizer = None  # инициализция ссылки на BatchVectorizer

batches_found = len(glob.glob(os.path.join(target_folder, '*.batch')))
if batches_found < 1:
    print "No batches found, parsing them from textual collection...",
    batch_vectorizer = artm.BatchVectorizer(data_path=train_data_folder,
                                            data_format='bow_uci',
                                            collection_name=collection_name,
                                            target_folder=target_folder)
    print " OK."
else:
    print "Found " + str(batches_found) + " batches, using them."
    batch_vectorizer = artm.BatchVectorizer(data_path=target_folder, data_format='batches')

model = artm.ARTM(num_topics=num_topics)

if not os.path.isfile(_dictionary_path):
    model.gather_dictionary(_dictionary_name, batch_vectorizer.data_path)
    model.save_dictionary(dictionary_name=_dictionary_name, dictionary_path=_dictionary_path)

model.load_dictionary(dictionary_name='dictionary', dictionary_path=_dictionary_path)
model.initialize(dictionary_name=_dictionary_name)

model.regularizers.add(artm.SmoothSparsePhiRegularizer(name='SmoothPsiRegularizer', class_ids=[labels_class]))
model.regularizers.add(artm.LabelRegularizationPhiRegularizer(name='LabelPsiRegularizer', class_ids=[labels_class]))
model.regularizers.add(artm.SmoothSparsePhiRegularizer(name='SmoothPhiRegularizer', class_ids=[tokens_class]))
model.regularizers.add(artm.SmoothSparseThetaRegularizer(name='SmoothThetaRegularizer'))

for iter in xrange(num_collection_passes):
    print 'Start processing iteration #' + str(iter) + '...'

    # Обновление значений коэфициентов регуляризации
    model.regularizers['SmoothPsiRegularizer'].tau = smooth_psi_tau[iter]
    model.regularizers['LabelPsiRegularizer'].tau = label_psi_tau[iter]
    model.regularizers['SmoothPhiRegularizer'].tau = smooth_phi_tau[iter]
    model.regularizers['SmoothThetaRegularizer'].tau = smooth_theta_tau[iter]

    # Обновление весов модальностей
    model.class_ids = {tokens_class: tokens_class_weight[iter], labels_class: labels_class_weight[iter]}

    # Обновление числа итераций прохода по документу
    model.num_document_passes = num_document_passes[iter]

    # Вызов метода обучения
    model.fit_offline(num_collection_passes=1, batch_vectorizer=batch_vectorizer)


