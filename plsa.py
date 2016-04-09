# coding: utf-8
__author__ = 'andrey'

# TODO Тут будет описание работы скрипта...

import os
import sys
import glob
import artm.library
import artm.messages_pb2
import shutil

# ===================================================
# Инициализация
# ===================================================
topics_num = 3  # число топиков для кластеризации

collection_name = 'banks'
data_folder = 'train/'
target_folder = 'plsa/'+collection_name  # целевая папка
csv_folder = 'csv'

logs_dir = 'logs'
bigartm_logs_dir = 'bigartm'  # BigArtm internal logs specific dir

# ===================================================
# Создание папок
# ===================================================

# создать папку logs_dir, если её нет
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# создать папку bigartm_logs_dir, если её нет
bigartm_logs_path = logs_dir + '/' + bigartm_logs_dir
if not os.path.exists(bigartm_logs_path):
    os.makedirs(bigartm_logs_path)

# создать папку, если её нет
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# удалить рекурсивно содержимое? папки
shutil.rmtree(csv_folder)
# создать папку, если её нет
if not os.path.exists(csv_folder):
    os.makedirs(csv_folder)

# ===================================================
# TODO Конфигурация парсера ОБУЧАЮЩЕЙ коллекции - реализовать в виде функции
# ===================================================

# идет проверка на наличие какого-то батча в папке
# TODO скорее всего batch это сохраненная модель или конфиг, позже выясним
batches_found = len(glob.glob(target_folder + "/*.batch"))
if batches_found != 0:
    shutil.rmtree(target_folder)

collection_parser_config = artm.messages_pb2.CollectionParserConfig()  # создание объекта конфигуратора
collection_parser_config.format = artm.library.CollectionParserConfig_Format_BagOfWordsUci  # формат мешка слов
# укзание на docword и vocab коллекции
collection_parser_config.docword_file_path = data_folder + 'docword.' + collection_name + '.txt'
collection_parser_config.vocab_file_path = data_folder + 'vocab.' + collection_name + '.txt'

collection_parser_config.target_folder = target_folder  # целевая папка

collection_parser_config.dictionary_file_name = 'dictionary'  # имя файла словаря

# Набор уникальных токенов
unique_tokens = artm.library.Library().ParseCollection(collection_parser_config)

print "Parsing from TRAIN textual collection: OK."

# sys.exit()  # stop execution

# ===================================================
# Построение тематической модели
# ===================================================

# Create master component and infer topic model
with artm.library.MasterComponent() as master:

    master.config().cache_theta = True
    master.Reconfigure()

    # Создание словаря BigArtm c частотами токенов
    dictionary = master.CreateDictionary(unique_tokens)

    # print unique_tokens

    # ===================================================
    # Конфигурирование базовых показателей score
    # ===================================================

    # Создаем один показатель топа токенов для каждой метки класса class_id
    ru_top_tokens_score = master.CreateTopTokensScore(class_id='@default_class')
    en_top_tokens_score = master.CreateTopTokensScore(class_id='@labels')

    theta_snippet_score = master.CreateThetaSnippetScore()

    # ===================================================
    # Конфигурирование модели
    # ===================================================

    # Populate class_id and class_weight in ModelConfig
    config = artm.messages_pb2.ModelConfig()
    config.class_id.append('@default_class')
    config.class_weight.append(0.10)
    config.class_id.append('@labels')
    config.class_weight.append(1.00)

    # ===================================================
    # Создание модели
    # ===================================================

    # Создание модели: задание числа топиков, числа внутренних итераций, применение конфигурации
    model = master.CreateModel(topics_count=topics_num, inner_iterations_count=10, config=config)
    # Инициализация модели
    model.Initialize(dictionary)  # Setup initial approximation for Phi matrix.
    # Включение показателей
    model.EnableScore(ru_top_tokens_score)
    model.EnableScore(en_top_tokens_score)
    model.EnableScore(theta_snippet_score)

    # ===================================================
    # Обучение модели (итерация)
    # ===================================================

    # Обучаем модель за 10 проходов
    for iteration in range(0, 10):
        # master.AddBatch(batch=batch)                   # Over the batch.
        master.InvokeIteration(disk_path=target_folder)  # Invoke one scan over all batches,
        master.WaitIdle()                                # wait for all batches are processed
        model.Synchronize()                              # synchronize model

    # Получение и визуализация топа токенов в каждом топике с учетом класса
    artm.library.Visualizers.PrintTopTokensScore(ru_top_tokens_score.GetValue(model))
    artm.library.Visualizers.PrintTopTokensScore(en_top_tokens_score.GetValue(model))
    #artm.library.Visualizers.PrintThetaSnippetScore(theta_snippet_score.GetValue(model))

    # ===================================================
    # Сохранение матрицы тета обучающей выборки (распеределение топиков по документу)
    # ===================================================

    theta_matrix = master.GetThetaMatrix(model, clean_cache=True)  # извлекаем тета-матрицу из модели
    f = open(csv_folder + '/' + 'theta_train.csv', 'w')            # создаем csv для сохранения

    for item in theta_matrix.item_weights:  # цикл по всем документам тета-матрицы
        str1 = ''
        # print item.value
        for val in item.value:                # цикл по всем топикам документа
          str1 = str1+str(val)+';'              # накапливаем строку, разделенную ";"
        f.write(str1 + '\n')                  # пишем строку в csv файл

    f.close()

    # ===================================================
    # Тестирование модели
    # ===================================================

    data_folder = 'test/'
    collection_name = 'banks'
    target_folder = 'plsa/'+collection_name

    # ===================================================
    # TODO Конфигурация парсера ТЕСТОВОЙ коллекции - функция
    # ===================================================

    # TODO опять таинственные батчи
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

    print "Parsing from TEST textual collection: OK."

    batches = glob.glob(target_folder + "/*.batch")

    test_batch = artm.library.Library().LoadBatch(batches[0])  # грузим с диска первый батч

    # Применяем модель к батчу и получаем тета-матрицу тестовой выборки
    theta_matrix = master.GetThetaMatrix(model=model, batch=test_batch)

    labels_file_path = data_folder + 'labels.' + collection_name + '.txt'
    with open(labels_file_path) as f:
        test_y = [line.rstrip() for line in f]  # массив меток для проверки теста
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

# Moving BigArtm internal logs to specific dir
for filename in os.listdir('.'):
    if os.path.isfile(filename) and filename.startswith('..'):
        # If you specify the full path to the destination (not just the directory)
        # then shutil.move will overwrite any existing file
        shutil.move(os.path.join('.', filename), os.path.join(bigartm_logs_path, filename))


# print 'STOP'
# sys.exit()  # stop execution