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

def print_measures(model_plsa, model_artm):
    print 'Sparsity Phi: {0:.3f} (PLSA) vs. {1:.3f} (ARTM)'.format(
        model_plsa.score_tracker['SparsityPhiScore'].last_value,
        model_artm.score_tracker['SparsityPhiScore'].last_value)

    print 'Sparsity Theta: {0:.3f} (PLSA) vs. {1:.3f} (ARTM)'.format(
        model_plsa.score_tracker['SparsityThetaScore'].last_value,
        model_artm.score_tracker['SparsityThetaScore'].last_value)

    print 'Kernel contrast: {0:.3f} (PLSA) vs. {1:.3f} (ARTM)'.format(
        model_plsa.score_tracker['TopicKernelScore'].last_average_contrast,
        model_artm.score_tracker['TopicKernelScore'].last_average_contrast)

    print 'Kernel purity: {0:.3f} (PLSA) vs. {1:.3f} (ARTM)'.format(
        model_plsa.score_tracker['TopicKernelScore'].last_average_purity,
        model_artm.score_tracker['TopicKernelScore'].last_average_purity)

    print 'Perplexity: {0:.3f} (PLSA) vs. {1:.3f} (ARTM)'.format(
        model_plsa.score_tracker['PerplexityScore'].last_value,
        model_artm.score_tracker['PerplexityScore'].last_value)

    plt.plot(xrange(model_plsa.num_phi_updates), model_plsa.score_tracker['PerplexityScore'].value, 'b--',
             xrange(model_artm.num_phi_updates), model_artm.score_tracker['PerplexityScore'].value, 'r--', linewidth=2)
    plt.xlabel('Iterations count')
    plt.ylabel('PLSA perp. (blue), ARTM perp. (red)')
    plt.grid(True)
    #plt.show()

collection_name = 'kos'
_dictionary_name = 'dictionary'
dictionary_filename = _dictionary_name + '.dict'
target_folder = os.path.join('target', collection_name)
train_data_folder = os.path.join('train', collection_name)  # папка с обучающей выборкой
_topics_count = 15           # 15 the best
_dictionary_path = os.path.join(target_folder, dictionary_filename )

_tau_phi = -0.1
_tau_decor = 1.5e+5
_tau_theta = -0.15


batch_vectorizer = None  # инициализция ссылки на BatchVectorizer

batches_found = len(glob.glob(os.path.join(target_folder, '*.batch')))
if batches_found < 1:
    print "No batches found, parsing them from textual collection...",
    batch_vectorizer = artm.BatchVectorizer(data_path=train_data_folder, data_format='bow_uci', collection_name=collection_name, target_folder=target_folder)
    print " OK."
else:
    print "Found " + str(batches_found) + " batches, using them."
    batch_vectorizer = artm.BatchVectorizer(data_path=target_folder, data_format='batches')


model_plsa = artm.ARTM(topic_names=['topic_{}'.format(i) for i in xrange(_topics_count)],
                  scores=[artm.PerplexityScore(name='PerplexityScore',
                                               use_unigram_document_model=False,
                                               dictionary_name=_dictionary_name)])

model_artm = artm.ARTM(topic_names=['topic_{}'.format(i) for i in xrange(_topics_count)],
                  scores=[artm.PerplexityScore(name='PerplexityScore',
                                               use_unigram_document_model=False,
                                               dictionary_name=_dictionary_name)],
                  regularizers=[artm.SmoothSparseThetaRegularizer(name='SparseTheta', tau=_tau_theta)])

if not os.path.isfile(_dictionary_path):
    model_plsa.gather_dictionary(_dictionary_name, batch_vectorizer.data_path)
    model_plsa.save_dictionary(dictionary_name=_dictionary_name, dictionary_path=_dictionary_path)

model_plsa.load_dictionary(dictionary_name=_dictionary_name, dictionary_path=_dictionary_path)
model_artm.load_dictionary(dictionary_name=_dictionary_name, dictionary_path=_dictionary_path)

model_plsa.initialize(dictionary_name=_dictionary_name)
model_artm.initialize(dictionary_name=_dictionary_name)

model_plsa.scores.add(artm.SparsityPhiScore(name='SparsityPhiScore'))
model_plsa.scores.add(artm.SparsityThetaScore(name='SparsityThetaScore'))
model_plsa.scores.add(artm.TopicKernelScore(name='TopicKernelScore', probability_mass_threshold=0.3))

model_artm.scores.add(artm.SparsityPhiScore(name='SparsityPhiScore'))
model_artm.scores.add(artm.SparsityThetaScore(name='SparsityThetaScore'))
model_artm.scores.add(artm.TopicKernelScore(name='TopicKernelScore', probability_mass_threshold=0.3))

model_artm.regularizers.add(artm.SmoothSparsePhiRegularizer(name='SparsePhi', tau=_tau_phi))
model_artm.regularizers.add(artm.DecorrelatorPhiRegularizer(name='DecorrelatorPhi', tau=_tau_decor))

model_plsa.num_document_passes = 1
model_artm.num_document_passes = 1
model_plsa.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=15)
model_artm.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=15)

print_measures(model_plsa, model_artm)

# update tau_coefficients of regularizers in Model
model_artm.regularizers['SparsePhi'].tau = -0.2
model_artm.regularizers['SparseTheta'].tau = -0.2
model_artm.regularizers['DecorrelatorPhi'].tau = 2.5e+5

model_plsa.scores.add(artm.TopTokensScore(name='TopTokensScore', num_tokens=6))
model_artm.scores.add(artm.TopTokensScore(name='TopTokensScore', num_tokens=6))

# дообучим модель
model_plsa.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=25)
model_artm.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=25)

print_measures(model_plsa, model_artm)

plt.plot(xrange(model_plsa.num_phi_updates), model_plsa.score_tracker['SparsityPhiScore'].value, 'b--',
                 xrange(model_artm.num_phi_updates), model_artm.score_tracker['SparsityPhiScore'].value, 'r--', linewidth=2)
plt.xlabel('Iterations count')
plt.ylabel('PLSA Phi sp. (blue), ARTM Phi sp. (red)')
plt.grid(True)
#plt.show()

plt.plot(xrange(model_plsa.num_phi_updates), model_plsa.score_tracker['SparsityThetaScore'].value, 'b--',
                 xrange(model_artm.num_phi_updates), model_artm.score_tracker['SparsityThetaScore'].value, 'r--', linewidth=2)
plt.xlabel('Iterations count')
plt.ylabel('PLSA Theta sp. (blue), ARTM Theta sp. (red)')
plt.grid(True)
#plt.show()

for topic_name in model_plsa.topic_names:
    print topic_name + ': ',
    print model_plsa.score_tracker['TopTokensScore'].last_topic_info[topic_name].tokens

for topic_name in model_artm.topic_names:
    print topic_name + ': ',
    print model_artm.score_tracker['TopTokensScore'].last_topic_info[topic_name].tokens

print model_artm.phi_

# SAVE & LOAD

# Save the model to disk
print 'Saving model...\n'
model_artm.save(filename=os.path.join(target_folder, 'kos_artm_model'))  # save the model to disk

# TODO Load saved model into new instance - надо все понять как это работает!
print 'Loading model...\n'
model = artm.ARTM(topic_names=[])         # create new model
model.load(filename=os.path.join(target_folder, 'kos_artm_model'))     # load saved model into new instance

theta_matrix = model_artm.get_theta()
print theta_matrix

# Test on test data
