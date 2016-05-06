# -*- coding: utf-8 -*-
"""Эксперимент по обработке данных UCI с мультимодальностью для BigARTM v 0.7.5

   Для начала коллекция представлена данными quant_OHSUMED_test_87.txt

   Как мне надо не работает

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
_dictionary_path = os.path.join(target_folder, dictionary_filename )


_topics_count = 3
_tau_phi = -0.1
#_tau_decor = 1.5e+5
#_tau_theta = -0.15

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
                       class_ids={'@default_class':1.00, '@class':1.00})

if not os.path.isfile(_dictionary_path):
    model_artm.gather_dictionary(_dictionary_name, batch_vectorizer.data_path)
    model_artm.save_dictionary(dictionary_name=_dictionary_name, dictionary_path=_dictionary_path)

model_artm.load_dictionary(dictionary_name=_dictionary_name, dictionary_path=_dictionary_path)
model_artm.initialize(dictionary_name=_dictionary_name)

model_artm.scores.add(artm.TopTokensScore(name='TopTokensScoreDefault', num_tokens=6, class_id='@default_class'))
model_artm.scores.add(artm.TopTokensScore(name='TopTokensScoreClass', num_tokens=6, class_id='@class'))

model_artm.scores.add(artm.SparsityPhiScore(name='SparsityPhiScoreDefault', class_id='@default_class'))
model_artm.scores.add(artm.SparsityPhiScore(name='SparsityPhiScoreClass', class_id='@class'))

model_artm.scores.add(artm.SparsityThetaScore(name='SparsityThetaScore')) # TODO почему то одна

model_artm.num_document_passes = 10 # TODO соотнести со старым кодом, хотя это есть на входе fit_offline
model_artm.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=8)


print model_artm.score_tracker['TopTokensScoreDefault'].last_topic_info['topic_0'].tokens

# Retrieve and visualize top tokens in each topic
print 'TopTokensScore default_class'
for topic_name in model_artm.topic_names:
    print topic_name + ': ',
    print model_artm.score_tracker['TopTokensScoreDefault'].last_topic_info[topic_name].tokens

print 'TopTokensScore class'
for topic_name in model_artm.topic_names:
    print topic_name + ': ',
    print model_artm.score_tracker['TopTokensScoreClass'].last_topic_info[topic_name].tokens

