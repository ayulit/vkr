# -*- coding: utf-8 -*-
"""Запуск примера из туториала BigARTM

   Сам пример на русском написан тут
   http://nbviewer.jupyter.org/github/bigartm/bigartm-book/blob/master/BigARTM_example_RU.ipynb
   Но в отличие от оригинала код адартирован к более старой версии BigARTM, а именно 0.7.1

   Requirements: BigArtm v 0.7.1

"""

import glob
import os
import matplotlib.pyplot as plt

import artm.messages_pb2
import artm.library
import artm.artm_model
from pandas import DataFrame

import sys

# Parse collection TODO Сделать prewash от всех каментов - они устарели по смыслу
collection_name = 'kos'
target_folder = os.path.join('target', collection_name)
train_data_folder = os.path.join('train', collection_name)  # папка с обучающей выборкой

_topics_count = 5           # 15 the best
_num_document_passes = 1    # 15 the best
_num_collection_passes = 5  # 15 the best

_tau_phi = -0.15
_tau_decor = 1.5e+5
_tau_theta = -0.2

perplexity_score_tracker = []
sparsity_phi_score_tracker = []
sparsity_theta_score_tracker = []

batches_found = len(glob.glob(os.path.join(target_folder, '*.batch')))
if batches_found == 0:
    print "No batches found, parsing them from textual collection...",
    collection_parser_config = artm.messages_pb2.CollectionParserConfig()
    collection_parser_config.format = artm.library.CollectionParserConfig_Format_BagOfWordsUci

    collection_parser_config.docword_file_path = os.path.join(train_data_folder, 'docword.' + collection_name + '.txt')
    collection_parser_config.vocab_file_path = os.path.join(train_data_folder, 'vocab.' + collection_name + '.txt')
    collection_parser_config.target_folder = target_folder
    collection_parser_config.dictionary_file_name = 'dictionary'

    artm.library.Library().ParseCollection(collection_parser_config)
    print " OK."
else:
    print "Found " + str(batches_found) + " batches, using them."

# Create master component and infer topic model
with artm.library.MasterComponent() as master:

    # Без этих 2-х строк тета-матрица будет пуста!
    master.config().cache_theta = True
    master.Reconfigure()

    # Create dictionary with tokens frequencies
    master.ImportDictionary('dictionary', os.path.join(target_folder, 'dictionary'))

    # Configure basic scores
    perplexity_score = master.CreatePerplexityScore()
    sparsity_phi_score = master.CreateSparsityPhiScore()
    sparsity_theta_score = master.CreateSparsityThetaScore()
    # theta_snippet_score = master.CreateThetaSnippetScore()


    # Configure basic regularizers
    smsp_phi_reg = master.CreateSmoothSparsePhiRegularizer()
    decorrelator_reg = master.CreateDecorrelatorPhiRegularizer()

    # Configure the model
    model = master.CreateModel(topics_count=_topics_count, inner_iterations_count=_num_document_passes)

    model.EnableRegularizer(smsp_phi_reg, _tau_phi)
    model.EnableRegularizer(decorrelator_reg, _tau_decor)

    model.Initialize('dictionary')       # Setup initial approximation for Phi matrix.

    _num_phi_updates = _num_collection_passes  # для совместимости с идеей описания
    for iteration in range(0, _num_phi_updates):
        master.InvokeIteration(disk_path=target_folder)  # Invoke one scan over all batches,
        master.WaitIdle()                                # and wait until it completes.
        model.Synchronize()                              # Synchronize topic model.

        print "Iter# :" + str(iteration),
        print " Perplexity = %.3f (ARTM), " % perplexity_score.GetValue(model).value,
        print " Sparsity Phi = %.3f (ARTM), " % sparsity_phi_score.GetValue(model).value,
        print " Sparsity Theta = %.3f (ARTM)" % sparsity_theta_score.GetValue(model).value

        perplexity_score_tracker.append(perplexity_score.GetValue(model).value)
        sparsity_phi_score_tracker.append(sparsity_phi_score.GetValue(model).value)
        sparsity_theta_score_tracker.append(sparsity_theta_score.GetValue(model).value)

    plt.plot(xrange(_num_phi_updates), perplexity_score_tracker, 'r--', linewidth=2)
    plt.xlabel('Iterations count')
    plt.ylabel('ARTM perp. (red)')
    plt.grid(True)
    #plt.show()




    # update tau_coefficients of regularizers in Model
    _tau_phi = -0.2
    _tau_decor = 2.5e+5
    config_copy = artm.messages_pb2.ModelConfig()
    config_copy.CopyFrom(model.config())
    config_copy.regularizer_tau[0] = _tau_phi  # 0, т.к. он был включен первым
    config_copy.regularizer_tau[1] = _tau_decor  # 1, т.к. он был включен вторым
    model.Reconfigure(config_copy)

    # Adding Smooth Sparse Theta regularizers
    smsp_theta_reg = master.CreateSmoothSparseThetaRegularizer()
    model.EnableRegularizer(smsp_theta_reg, _tau_theta)

    # Configure Top Tokens scores with 6 top tokens
    top_tokens_score = master.CreateTopTokensScore(num_tokens=6)

    _num_collection_passes = 25
    for iteration in range(_num_phi_updates, _num_phi_updates + _num_collection_passes):
        master.InvokeIteration(disk_path=target_folder)  # Invoke one scan over all batches,
        master.WaitIdle()                                # and wait until it completes.
        model.Synchronize()                              # Synchronize topic model.

        print "Iter# :" + str(iteration),
        print " Perplexity = %.3f (ARTM), " % perplexity_score.GetValue(model).value,
        print " Sparsity Phi = %.3f (ARTM), " % sparsity_phi_score.GetValue(model).value,
        print " Sparsity Theta = %.3f (ARTM)" % sparsity_theta_score.GetValue(model).value

        perplexity_score_tracker.append(perplexity_score.GetValue(model).value)
        sparsity_phi_score_tracker.append(sparsity_phi_score.GetValue(model).value)
        sparsity_theta_score_tracker.append(sparsity_theta_score.GetValue(model).value)

    _num_phi_updates += _num_collection_passes
    plt.plot(xrange(_num_phi_updates), perplexity_score_tracker, 'r--', linewidth=2)
    plt.xlabel('Iterations count')
    plt.ylabel('ARTM perp. (red)')
    plt.grid(True)
    #plt.show()

    plt.plot(xrange(_num_phi_updates), sparsity_phi_score_tracker, 'r--', linewidth=2)
    plt.xlabel('Iterations count')
    plt.ylabel('ARTM Phi sp. (red)')
    plt.grid(True)
    #plt.show()

    plt.plot(xrange(_num_phi_updates), sparsity_theta_score_tracker, 'r--', linewidth=2)
    plt.xlabel('Iterations count')
    plt.ylabel('ARTM Theta sp. (red)')
    plt.grid(True)
    #plt.show()

    artm.library.Visualizers.PrintTopTokensScore(top_tokens_score.GetValue(model))
    # artm.library.Visualizers.PrintThetaSnippetScore(theta_snippet_score.GetValue(model))


    # Get Phi Matrix. Способ 1, наглядный

    print '\nGOOD PHI\n'

    topic_model, numpy_matrix  = master.GetTopicModel(model=model)
    tokens = [token for token in topic_model.token]
    topic_names = [topic_name for topic_name in topic_model.topic_name]

    retval = DataFrame(data=numpy_matrix,
                       columns=topic_names,
                       index=tokens)

    print retval

    # Get Phi Matrix. Способ 2, мутный

    # The following code retrieves one topic at a time.
    # This avoids retrieving large topic models in a single protobuf message.
    print "\nOutput p(w|t) values for the first few tokens (alphabetically) in each topic:\n"
    #topic_model, numpy_matrix = master.GetTopicModel(model=model)

    for topic_name in topic_names:
        topic_model, numpy_matrix = master.GetTopicModel(model=model, topic_names={topic_name})  # retrieve one column in Phi matrix
        print topic_model.topic_name[0],
        for i in range(0, 5):
            print topic_model.token[i], "%.5f" % numpy_matrix[i, 0],
        print "..."

    # Работает странно, надо ставить последнюю версию
    if False:

        # Save the model to disk
        print 'Saving topic model...\n'
        with open(os.path.join(target_folder, 'Output.topic_model'), 'wb') as binary_file:
            binary_file.write(master.GetTopicModel(model, use_matrix=False).SerializeToString())

        # Load saved model into new instance
        print 'Loading topic model...\n'
        topic_model = artm.messages_pb2.TopicModel()
        with open(os.path.join(target_folder, 'Output.topic_model'), 'rb') as binary_file:
            topic_model.ParseFromString(binary_file.read())

        awaken_model = master.CreateModel(topics_count=_topics_count, inner_iterations_count=_num_document_passes)
        awaken_model.Overwrite(topic_model)  # restore previously saved topic model into awaken_model



    # Get Theta Matrix
    print '\nGET THETA\n'

    theta_matrix, numpy_matrix = master.GetThetaMatrix(model, clean_cache=False)  # 0.7.1, clean_cache = remove_theta from RAM
    document_ids = [item_id for item_id in theta_matrix.item_id]

    # Getting inverted theta matrix
    #retval = DataFrame(data=theta_matrix[1].transpose(),
    #                   columns=document_ids,
    #                   index=topic_names)

    retval = DataFrame(data=numpy_matrix,
                       columns=topic_names,
                       index=document_ids)

    print retval

