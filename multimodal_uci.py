# -*- coding: utf-8 -*-
"""Эксперимент по обработке данных UCI с мультимодальностью

   Для начала коллекция представлена данными quant_OHSUMED_test_87.txt

"""

import os
import sys
import glob
import artm.library
import artm.messages_pb2
import shutil


# ===================================================
# Инициализация
# ===================================================

collection_name = 'ohsumed'                                # название коллекции
train_data_folder = os.path.join('train',collection_name)  # папка с обучающей выборкой
target_folder = os.path.join('target',collection_name)

_topics_count = 3

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

    with open(collection_parser_config.vocab_file_path) as f:  # открываем файл словаря коллекции
        index = 0
        for line in f:  #  цикл по всем словам
          if (index < 1000):
              collection_parser_config.cooccurrence_token.append(line.rstrip())
              index += 1
    collection_parser_config.gather_cooc = True

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
    # Create dictionary with tokens frequencies
    master.ImportDictionary('dictionary', os.path.join(target_folder, 'dictionary'))

    # ===================================================
    # Конфигурирование базовых показателей score
    # ===================================================

    # Create one top-token score per each class_id (модальности)
    default_top_tokens_score = master.CreateTopTokensScore(class_id='@default')    # топ токенов документа
    alpha_top_tokens_score = master.CreateTopTokensScore(class_id='@class')        # топ диагнозов

    default_sparsity = master.CreateSparsityPhiScore(class_id='@default')
    alpha_sparsity = master.CreateSparsityPhiScore(class_id='@class')

    sparsity_theta_score = master.CreateSparsityThetaScore()
    theta_snippet_score = master.CreateThetaSnippetScore()

    #perplexity_score = master.CreatePerplexityScore()
    #sparsity_phi_score = master.CreateSparsityPhiScore()
    #top_tokens_score = master.CreateTopTokensScore()



    # Configure basic regularizers
    #smsp_theta_reg = master.CreateSmoothSparseThetaRegularizer()
    #smsp_phi_reg = master.CreateSmoothSparsePhiRegularizer()
    #decorrelator_reg = master.CreateDecorrelatorPhiRegularizer()


    # ===================================================
    # Создание модели
    # ===================================================

    # Configure the model
    model = master.CreateModel(topics_count=_topics_count, inner_iterations_count=10,
                               class_ids=('@default', '@class'),
                               class_weights=(1.00, 1.00))
    #model.EnableRegularizer(smsp_theta_reg, -0.1)
    #model.EnableRegularizer(smsp_phi_reg, -0.2)
    #model.EnableRegularizer(decorrelator_reg, 1000000)
    model.Initialize('dictionary')       # Setup initial approximation for Phi matrix.

    # Infer the model in 10 passes over the batch
    for iteration in range(0, 8):
        master.InvokeIteration(disk_path=target_folder)  # Invoke one scan over all batches,
        master.WaitIdle()                                # and wait until it completes.
        model.Synchronize()                              # Synchronize topic model.

        #print "Iter#" + str(iteration),
        #print ": Perplexity = %.3f" % perplexity_score.GetValue(model).value,
        #print ", Phi sparsity = %.3f" % sparsity_phi_score.GetValue(model).value,
        #print ", Theta sparsity = %.3f" % sparsity_theta_score.GetValue(model).value


    # Retrieve and visualize top tokens in each topic
    artm.library.Visualizers.PrintTopTokensScore(default_top_tokens_score.GetValue(model))
    artm.library.Visualizers.PrintTopTokensScore(alpha_top_tokens_score.GetValue(model))
    artm.library.Visualizers.PrintThetaSnippetScore(theta_snippet_score.GetValue(model))


