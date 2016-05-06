# -*- coding: utf-8 -*-
"""Сравнение мультимодальной модели со стандартным тематическим моделированием PLSA

   Строится график PLSA perp. (blue), ARTM perp. (red)

"""

import os
import sys
import glob
import artm.library
import artm.messages_pb2
import shutil
import matplotlib.pyplot as plt


# ===================================================
# Инициализация
# ===================================================

collection_name = 'ohsumed'                                # название коллекции
labels_class = '@labels'
train_data_folder = os.path.join('train',collection_name)  # папка с обучающей выборкой
target_folder = os.path.join('target',collection_name)

_topics_count = 88

# ===================================================
# Конфигурация парсера ОБУЧАЮЩЕЙ коллекции
# ===================================================

batches_found = len(glob.glob(os.path.join(target_folder,"*.batch")))
if batches_found == 0:
    print "No batches found, parsing them from textual collection...",
    collection_parser_config = artm.messages_pb2.CollectionParserConfig()
    collection_parser_config.format = artm.library.CollectionParserConfig_Format_BagOfWordsUci

    collection_parser_config.docword_file_path = os.path.join(train_data_folder,'docword.' + collection_name + '.txt')
    collection_parser_config.vocab_file_path = os.path.join(train_data_folder,'vocab.' + collection_name + '.txt')
    collection_parser_config.target_folder = target_folder
    collection_parser_config.dictionary_file_name = 'dictionary'

    # это dictionary
    unique_tokens = artm.library.Library().ParseCollection(collection_parser_config)

    print "unique_tokens=", unique_tokens
    print " OK."
else:
    print "Found " + str(batches_found) + " batches, using them."

# ===================================================
# Построение и запуск тематической модели
# ===================================================

# Create master component and infer topic model
with artm.library.MasterComponent() as master:

    # Без этих 2-х строк тета-матрица будет пуста!
    master.config().cache_theta = True
    master.Reconfigure()


    # Create dictionary with tokens frequencies
    master.ImportDictionary('dictionary', os.path.join(target_folder, 'dictionary'))


    # ===================================================
    # Создание модели
    # ===================================================

    # Configure the model

    model_plsa = master.CreateModel(topics_count=_topics_count, inner_iterations_count=1)


    model_artm = master.CreateModel(topics_count=_topics_count, inner_iterations_count=1,
                               class_ids=('@default_class', labels_class),
                               class_weights=(1.00, 1.00))

    model_plsa.Initialize('dictionary')
    model_artm.Initialize('dictionary')


    # ===================================================
    # Конфигурирование базовых показателей score
    # ===================================================

    perplexity_score_artm = master.CreatePerplexityScore(class_ids=['@default_class',labels_class])
    perplexity_score_plsa = master.CreatePerplexityScore(class_ids=['@default_class'])

    # Create one top-token score per each class_id (модальности)
    #default_top_tokens_score = master.CreateTopTokensScore(class_id='@default_class')  # топ токенов документа
    #alpha_top_tokens_score = master.CreateTopTokensScore(class_id=labels_class)        # топ диагнозов

    #default_sparsity = master.CreateSparsityPhiScore(class_id='@default_class')
    #alpha_sparsity = master.CreateSparsityPhiScore(class_id=labels_class)

    #sparsity_theta_score = master.CreateSparsityThetaScore()
    #theta_snippet_score = master.CreateThetaSnippetScore()

    #sparsity_phi_score = master.CreateSparsityPhiScore()
    #top_tokens_score = master.CreateTopTokensScore()

    # Configure basic regularizers
    smsp_theta_reg = master.CreateSmoothSparseThetaRegularizer()
    smsp_phi_reg = master.CreateSmoothSparsePhiRegularizer(class_ids=['@default_class',labels_class])
    decorrelator_reg = master.CreateDecorrelatorPhiRegularizer(class_ids=['@default_class',labels_class])



    model_artm.EnableRegularizer(smsp_theta_reg, tau=-0.15)
    #model_artm.EnableRegularizer(smsp_phi_reg, -0.2)
    #model_artm.EnableRegularizer(decorrelator_reg, 1000000)


    perplexity_score_plsa_tracker = []
    perplexity_score_artm_tracker = []

    # Infer the model in 10 passes over the batch
    _num_phi_updates = 9
    for iteration in range(0, _num_phi_updates):
        master.InvokeIteration(disk_path=target_folder)  # Invoke one scan over all batches,
        master.WaitIdle()                                # and wait until it completes.
        model_plsa.Synchronize()                         # Synchronize topic model.
        model_artm.Synchronize()                         # Synchronize topic model.

        print "Iter#" + str(iteration),
        print 'Perplexity: {0:.3f} (PLSA) vs. {1:.3f} (ARTM)'.format(
            perplexity_score_plsa.GetValue(model_plsa).value,
            perplexity_score_artm.GetValue(model_artm).value)
        #print ": Perplexity = %.3f" % perplexity_score_artm.GetValue(model_artm).value
        #print ", Phi sparsity = %.3f" % sparsity_phi_score.GetValue(model).value,
        #print ", Theta sparsity = %.3f" % sparsity_theta_score.GetValue(model).value

        perplexity_score_plsa_tracker.append(perplexity_score_plsa.GetValue(model_plsa).value)
        perplexity_score_artm_tracker.append(perplexity_score_artm.GetValue(model_artm).value)


    plt.plot(xrange(1,_num_phi_updates), perplexity_score_plsa_tracker[1:], 'b--',
             xrange(1,_num_phi_updates), perplexity_score_artm_tracker[1:], 'r--', linewidth=2)
    plt.xlabel('Iterations count')
    plt.ylabel('PLSA perp. (blue), ARTM perp. (red)')
    plt.grid(True)
    plt.show()

    # Retrieve and visualize top tokens in each topic
    #print 'default_class'
    #artm.library.Visualizers.PrintTopTokensScore(default_top_tokens_score.GetValue(model))
    #print labels_class
    #artm.library.Visualizers.PrintTopTokensScore(alpha_top_tokens_score.GetValue(model))
    #artm.library.Visualizers.PrintThetaSnippetScore(theta_snippet_score.GetValue(model))


