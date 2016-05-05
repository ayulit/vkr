# -*- coding: utf-8 -*-
"""Запуск примера из туториала BigARTM 0.7.5

   Сам пример на русском написан тут
   http://nbviewer.jupyter.org/github/bigartm/bigartm-book/blob/master/BigARTM_example_RU.ipynb

   Requirements: BigArtm v 0.7.5

"""

import glob
import os
import matplotlib.pyplot as plt

import artm




collection_name = 'kos'
_dictionary_name = 'dictionary'
dictionary_filename = _dictionary_name + '.dict'
target_folder = os.path.join('target', collection_name)
train_data_folder = os.path.join('train', collection_name)  # папка с обучающей выборкой
_topics_count = 3           # 15 the best
_dictionary_path = os.path.join(target_folder, dictionary_filename )

_tau_phi = -0.1
_tau_decor = 1.5e+5
_tau_theta = -0.15

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


model_artm = artm.ARTM(topic_names=['topic_{}'.format(i) for i in xrange(_topics_count)],
                  scores=[artm.PerplexityScore(name='PerplexityScore',
                                               use_unigram_document_model=False,
                                               dictionary_name=_dictionary_name)])

if not os.path.isfile(_dictionary_path):
    model_artm.gather_dictionary(_dictionary_name, batch_vectorizer.data_path)
    model_artm.save_dictionary(dictionary_name=_dictionary_name, dictionary_path=_dictionary_path)

model_artm.load_dictionary(dictionary_name=_dictionary_name, dictionary_path=_dictionary_path)
model_artm.initialize(dictionary_name=_dictionary_name)

model_artm.scores.add(artm.TopTokensScore(name='TopTokensScore', num_tokens=6))
model_artm.scores.add(artm.SparsityPhiScore(name='SparsityPhiScore'))
model_artm.scores.add(artm.SparsityThetaScore(name='SparsityThetaScore'))

model_artm.num_document_passes = 10
model_artm.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=8)

# Retrieve and visualize top tokens in each topic
print 'TopTokensScore default_class'
for topic_name in model_artm.topic_names:
    print topic_name + ': ',
    print model_artm.score_tracker['TopTokensScore'].last_topic_info[topic_name].tokens


