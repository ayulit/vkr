# coding: utf-8
__author__ = 'andrey'

import os
import sys
import glob
import artm.library
import artm.messages_pb2
import shutil

collection_name = 'banks'
data_folder = 'train/'
target_folder = 'plsa/'+collection_name
csv_folder = 'csv'

# создать папку, если её нет
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# удалить рекурсивно содержимое? папки
shutil.rmtree(csv_folder)
# создать папку, если её нет
if not os.path.exists(csv_folder):
    os.makedirs(csv_folder)

# идет проверка на наличие какого-то батча в папке
batches_found = len(glob.glob(target_folder + "/*.batch"))

if batches_found != 0:
    shutil.rmtree(target_folder)

collection_parser_config = artm.messages_pb2.CollectionParserConfig()
collection_parser_config.format = artm.library.CollectionParserConfig_Format_BagOfWordsUci

collection_parser_config.docword_file_path = data_folder + 'docword.' + collection_name + '.txt'
collection_parser_config.vocab_file_path = data_folder + 'vocab.' + collection_name + '.txt'

collection_parser_config.target_folder = target_folder

collection_parser_config.dictionary_file_name = 'dictionary'

unique_tokens = artm.library.Library().ParseCollection(collection_parser_config)

print "Parsing from textual collection: OK."

# Create master component and infer topic model
with artm.library.MasterComponent() as master:

    master.config().cache_theta = True
    master.Reconfigure()

    # Create dictionary with tokens frequencies
    dictionary = master.CreateDictionary(unique_tokens)

    #print unique_tokens

    # Configure basic scores

    # Create one top-token score per each class_id
    ru_top_tokens_score = master.CreateTopTokensScore(class_id='@default_class')
    en_top_tokens_score = master.CreateTopTokensScore(class_id='@labels')

    theta_snippet_score  = master.CreateThetaSnippetScore()


    # Populate class_id and class_weight in ModelConfig
    config = artm.messages_pb2.ModelConfig()
    config.class_id.append('@default_class')
    config.class_weight.append(0.10)
    config.class_id.append('@labels')
    config.class_weight.append(1.00)

    # Configure the model

    # Create and initialize model, enable scores. Our expert knowledge says we need 2 topics ;)
    model = master.CreateModel(topics_count=3, inner_iterations_count=10, config=config)
    model.Initialize(dictionary)  # Setup initial approximation for Phi matrix.
    model.EnableScore(ru_top_tokens_score)
    model.EnableScore(en_top_tokens_score)

    model.EnableScore(theta_snippet_score)

    # Infer the model in 10 passes over the batch
    for iteration in range(0, 10):
        # master.AddBatch(batch=batch)
        master.InvokeIteration(disk_path=target_folder)
        master.WaitIdle()  # wait for all batches are processed
        model.Synchronize()  # synchronize model

    # Retrieve and visualize top tokens in each topic
    artm.library.Visualizers.PrintTopTokensScore(ru_top_tokens_score.GetValue(model))
    artm.library.Visualizers.PrintTopTokensScore(en_top_tokens_score.GetValue(model))
    #artm.library.Visualizers.PrintThetaSnippetScore(theta_snippet_score.GetValue(model))

    theta_matrix = master.GetThetaMatrix(model, clean_cache=True)
    f = open(csv_folder + '/' + 'theta_train.csv', 'w')
    # Retrieve and visualize scores
    for item in theta_matrix.item_weights:  # цикл по всем документам тета-матрицы
        str1 = ''
        #print item.value
        for val in item.value:  # цикл по всем топикам документа
          str1 = str1+str(val)+';'  # накапливаем строку, разделенную ";"
        f.write(str1 + '\n')  # пишем строку в файл
    f.close()

    #########
    data_folder = 'test/'
    collection_name = 'banks'
    target_folder = 'plsa/'+collection_name

    batches_found = len(glob.glob(target_folder + "/*.batch"))

    if batches_found != 0:
        shutil.rmtree(target_folder)

    collection_parser_config = artm.messages_pb2.CollectionParserConfig()
    collection_parser_config.format = artm.library.CollectionParserConfig_Format_BagOfWordsUci
    collection_parser_config.docword_file_path = data_folder + 'docword.' + collection_name + '.txt'
    collection_parser_config.vocab_file_path = data_folder + 'vocab.' + collection_name + '.txt'
    collection_parser_config.target_folder = target_folder
    collection_parser_config.dictionary_file_name = 'dictionary'
    unique_tokens = artm.library.Library().ParseCollection(collection_parser_config)

    batches = glob.glob(target_folder + "/*.batch")



    test_batch = artm.library.Library().LoadBatch(batches[0])
    theta_matrix = master.GetThetaMatrix(model=model, batch=test_batch)

    labels_file_path = data_folder + 'labels.' + collection_name + '.txt'
    with open(labels_file_path) as f:
        test_y = [line.rstrip() for line in f]
        f.close()
    # print "len test_y = ", len(test_y)
    # print test_y



    f = open(csv_folder + '/' + 'theta_test.csv', 'w')
    # Retrieve and visualize scores

    match_counter = 0

    for j, item in enumerate(theta_matrix.item_weights):  # цикл по всем документам тета-матрицы

        str1 = ''
        docvector=[]
        #print item.value
        for val in item.value:  # цикл по всем топикам документа
            docvector.append(val)
            str1 = str1+str(val)+';'  # накапливаем строку, разделенную ";"

        if test_y[j] == 'label1':
            real = 1
        elif test_y[j] == 'label2':
            real = 2
        elif test_y[j] == 'label0':
            real = 0


        str1 = str1+str(real)+';'

        predict = docvector.index(max(docvector))
        str1 = str1+str(predict)+';'

        if real == predict:
            match_counter += 1

        f.write(str1 + '\n')  # пишем строку в файл

    match = match_counter*1.0/len(theta_matrix.item_weights)
    f.write(str(match)+';' + '\n')
    f.close()

    shutil.rmtree(target_folder)



# print 'STOP'
# sys.exit()  # stop execution