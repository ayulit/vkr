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
# Описание функций
# ===================================================

def collectionParserConfig(data_folder):
    """Парсит данные коллекции и сохраняет их на HDD в специальном формате BigARTM

    Args:
        data_folder (basestring): Путь к данным.

    Returns:
        artm.messages_pb2.DictionaryConfig: Returns an instance of DictionaryConfig
        which carry all unique words in the collection and their frequencies.

    """
    batches_found = len(glob.glob(batches_pathname))
    if batches_found != 0:
        shutil.rmtree(target_folder)

    print "Parsing batches from textual collection...",

    collection_parser_config = artm.messages_pb2.CollectionParserConfig()
    collection_parser_config.format = artm.library.CollectionParserConfig_Format_BagOfWordsUci

    collection_parser_config.docword_file_path = os.path.join(data_folder,'docword.' + collection_name + '.txt')
    collection_parser_config.vocab_file_path = os.path.join(data_folder,'vocab.' + collection_name + '.txt')
    collection_parser_config.target_folder = target_folder
    collection_parser_config.dictionary_file_name = 'dictionary'

    # получаем объект коллекции BigArtm: тут будет None в 0.7.1 по крайней мере
    # так же метод ParseCollection сохраняет в target_folder на HDD batch и dictionary
    uniqueTokens = artm.library.Library().ParseCollection(collection_parser_config)

    print " OK."

    return uniqueTokens


# ===================================================
# Инициализация
# ===================================================
topics_num = 3                                           # число топиков для кластеризации

collection_name = 'banks'                                # название коллекции
train_data_folder = 'train'                              # папка с обучающей выборкой
test_data_folder = 'test'                                # папка с тестовой выборкой
target_folder = os.path.join('plsa',collection_name)     # папка с данными BigArtm: батч коллекции и словарь
csv_folder = 'csv'                                       # папка для csv файлов матриц тета и фи
logs_dir = 'logs'                                        # папка с логами
bigartm_logs_dir = 'bigartm'                             # BigArtm internal logs specific dir
batches_pathname = os.path.join(target_folder,"*.batch")

# ===================================================
# Создание папок
# ===================================================

# создать папку logs_dir, если её нет
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

bigartm_logs_path = os.path.join(logs_dir,bigartm_logs_dir)
if not os.path.exists(bigartm_logs_path):
    os.makedirs(bigartm_logs_path)

if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# удалить рекурсивно содержимое? папки
shutil.rmtree(csv_folder)
# а потом создать папку
if not os.path.exists(csv_folder):
    os.makedirs(csv_folder)

# Moving BigArtm internal logs to specific dir
for filename in os.listdir('./'):
    if os.path.isfile(filename) and filename.startswith('..'):
        # If you specify the full path to the destination (not just the directory)
        # then shutil.move will overwrite any existing file
        shutil.move(os.path.join('./', filename), os.path.join(bigartm_logs_path, filename))

# для надежности принудидельно удалим symlink на лог
# это происходит походу только в Linux
link_name = '..INFO'
if os.path.islink(link_name):
    os.unlink(link_name)

# ===================================================
# Конфигурация парсера ОБУЧАЮЩЕЙ коллекции
# ===================================================

# Набор уникальных токенов
unique_tokens = collectionParserConfig(train_data_folder)
print "\nParsing from TRAIN textual collection: OK.\n"

# ===================================================
# Построение тематической модели
# ===================================================

# Create master component and infer topic model
with artm.library.MasterComponent() as master:

    master.config().cache_theta = True
    master.Reconfigure()

    # Создание словаря BigArtm c частотами токенов
    #dictionary = master.CreateDictionary(unique_tokens) # oblsolete?
    master.ImportDictionary('dictionary', os.path.join(target_folder, 'dictionary'))  # 0.7.1

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
    #model.Initialize(dictionary)   # Setup initial approximation for Phi matrix. Obsolete?
    model.Initialize('dictionary')  # 0.7.1
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

    # Вывод топа ТОКЕНОВ в каждом топике
    artm.library.Visualizers.PrintTopTokensScore(ru_top_tokens_score.GetValue(model))
    print "\n"
    # Вывод топа КЛАССОВ в каждом топике
    artm.library.Visualizers.PrintTopTokensScore(en_top_tokens_score.GetValue(model))
    # TODO Вывод ... чего?
    #artm.library.Visualizers.PrintThetaSnippetScore(theta_snippet_score.GetValue(model))

    # ===================================================
    # Сохранение тета-матрицы (распеределение топиков по документу) ОБУЧАЮЩЕЙ выборки
    # ===================================================

    # извлекаем тета-матрицу из модели
    theta_matrix, numpy_matrix = master.GetThetaMatrix(model, clean_cache=True)  # 0.7.1
    # создаем csv для сохранения
    f = open(os.path.join(csv_folder,'theta_train.csv'), 'w')

    for item in numpy_matrix:      # цикл по всем документам тета-матрицы. 0.7.1
        str1 = ''
        # print item.value
        for val in item:             # цикл по всем топикам документа
          str1 = str1+str(val)+';'     # накапливаем строку, разделенную ";"
        f.write(str1 + '\n')         # пишем строку в csv файл

    f.close()

    # ===================================================
    # Тестирование модели
    # ===================================================

    # ===================================================
    # Конфигурация парсера ТЕСТОВОЙ коллекции
    # ===================================================

    # получаем объект коллекции BigArtm
    # так же метод ParseCollection сохраняет в target_folder на HDD batch и dictionary
    unique_tokens = collectionParserConfig(test_data_folder)
    print "\n\nParsing from TEST textual collection: OK.\n"

    batches = glob.glob(batches_pathname)  # считываем все батчи с HDD в список

    # берем первый батч, т.к. считаем, что в target_folder больше нет,
    # т.е. первый батч и будет нужной тестовой коллекцией в формате BigArtm
    test_batch = artm.library.Library().LoadBatch(batches[0])

    # Применяем модель к батчу и получаем тета-матрицу тестовой выборки
    theta_matrix, numpy_matrix = master.GetThetaMatrix(model=model, batch=test_batch)  # 0.7.1

    # путь к файлу с ручной разметкой тестовой коллекции
    labels_file_path = os.path.join(test_data_folder,'labels.' + collection_name + '.txt')
    with open(labels_file_path) as f:
        test_y = [line.rstrip() for line in f]  # записывае метки для проверки теста в список
        f.close()

    # print "len test_y = ", len(test_y)
    # print test_y

    # ===================================================
    # Сохранение тета-матрицы (распеределение топиков по документу) ТЕСТОВОЙ выборки
    # ===================================================

    f = open(os.path.join(csv_folder,'theta_test.csv'), 'w')

    match_counter = 0                                       # счетчик сопадений предсказанной и истинной меток

    for j, item in enumerate(numpy_matrix):    # цикл по всем документам тета-матрицы. 0.7.1

        str1 = ''
        docvector=[]                             # вектор документа тета-матрицы
        #print item.value
        for val in item:                         # цикл по всем топикам документа. 0.7.1
            docvector.append(val)                  # формируем вектор документа
            str1 = str1+str(val)+';'               # накапливаем строку, разделенную ";"

        if test_y[j] == 'label1':
            real = 1
        elif test_y[j] == 'label2':
            real = 2
        elif test_y[j] == 'label0':
            real = 0

        str1 = str1+str(real)+';'                # добавляем в строку значение истинной метки класса
        # TODO предсказанная метка - индекс max значения вектора ?!? Понять почему?
        predict = docvector.index(max(docvector))
        str1 = str1+str(predict)+';'             # добавляем в строку значение ПРЕДСКАЗАННОЙ метки класса

        if real == predict:
            match_counter += 1                     # при совпадении меток приращаем счетчик совпадений
        f.write(str1 + '\n')                     # пишем накопленную строку в файл

    match = match_counter*1.0/len(numpy_matrix)  # считаем процент совпадений. 0.7.1
    f.write(str(match)+';' + '\n')               # и выводим % совпадений в последней строке
    f.close()

    shutil.rmtree(target_folder)                 # удаляем все из целевой папки


# print 'STOP'
# sys.exit()  # stop execution