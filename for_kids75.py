# -*- coding: utf-8 -*-
"""Туториал для BigARTM v 0.7.5

   Коллекция представлена данными коллекции banks

   На основе туториала
   "BigARTM. Примеры обучения моделей на Python для самых маленьких"
   http://nbviewer.jupyter.org/github/bigartm/bigartm-book/blob/master/BigARTM_basic_tutorial_RU.ipynb

"""

import artm
import os
import glob
import matplotlib.pyplot as plt

collection_name = 'banks'
train_data_folder = os.path.join('train', collection_name)
target_folder = os.path.join('target', collection_name)
dictionary_name = 'dictionary'
dictionary_filename = dictionary_name + '.dict'
dictionary_path = os.path.join(target_folder, dictionary_filename)

batch_vectorizer = None
batches_found = len(glob.glob(os.path.join(target_folder, '*.batch')))

# ===================================================
# ВВЕДЕНИЕ
# ===================================================

if batches_found < 1:
    print "No batches found, parsing them from textual collection...",
    # создаем из UCI батчи и сохраняем их в traget folder
    batch_vectorizer = artm.BatchVectorizer(data_path=train_data_folder,
                                            data_format='bow_uci',
                                            collection_name=collection_name,
                                            target_folder=target_folder)
    print " OK."
else:
    print "Found " + str(batches_found) + " batches, using them."
    # создаем объект BatchVectorizer на основе имеющихся батчей
    batch_vectorizer = artm.BatchVectorizer(data_path=target_folder, data_format='batches')

# как ни уверяют разработчики в туториале, создать просто так объект Dictionary тут не получится,
# поэтому перед созданием этого объекта зачем то нужно создавать хоть какую, но модель
model_for_dic = artm.ARTM()

# и только потом создаем словарь, если его нет
# vocab_file подсовываем для соблюдения порядка в словаре bigartm
if not os.path.isfile(dictionary_path):
    # собираем словарь
    model_for_dic.gather_dictionary(dictionary_target_name=dictionary_name,
                                    data_path=target_folder,
                                    vocab_file_path=os.path.join(train_data_folder, 'vocab.' + collection_name + '.txt'))
    # сохраняем словарь в бинарном виде (автоматом подставляется расширение .dict)
    model_for_dic.save_dictionary(dictionary_name=dictionary_name, dictionary_path=dictionary_path)

# ===================================================
# РАЗДЕЛ 1: Обучение базовой модели PLSA с подсчётом перплексии
# ===================================================

# Создаем модель: пока тупо набор настроек
model = artm.ARTM(num_topics=20)
# Подгружаем словарь bigartm в модель: теперь из набора настроек возникает нулевая матрица Фи
model.load_dictionary(dictionary_name=dictionary_name, dictionary_path=dictionary_path)
# Инициализируем модель словарем bigartm (просто по уникальному имени!):
# заполняем матрицу Фи случайными значениями
model.initialize(dictionary_name=dictionary_name)

# Подключаем модели метрику перплексии (метрика - как объект)
model.scores.add(artm.PerplexityScore(name='my_fisrt_perplexity_score',
                                      use_unigram_document_model=False,
                                      dictionary_name=dictionary_name))

# Обучение модели оффлайновым алгоритмом:
#  - много проходов по коллекции (num_collection_passes)
#  - один (опционально) проход по документу (num_document_passes)
#  - обновление Фи в конце каждого прохода по коллекции
model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=10)

# Динамика значений перплексии за проходы
print model.score_tracker['my_fisrt_perplexity_score'].value

# График динамики перплексии
plt.plot(xrange(model.num_phi_updates), model.score_tracker['my_fisrt_perplexity_score'].value, 'b--')
plt.xlabel('Iterations count')
plt.ylabel('PLSA perp. (blue)')
plt.grid(True)
#plt.show()

# Дообучим модель, установив 5 проходов по документу и сделав еще 15 итераций по коллекции
model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=15, num_document_passes=5)

# Динамика значений перплексии за проходы
print model.score_tracker['my_fisrt_perplexity_score'].value

# График динамики перплексии (внимание: счетчик num_phi_updates не сбрасывается!)
plt.plot(xrange(model.num_phi_updates), model.score_tracker['my_fisrt_perplexity_score'].value, 'b--')
plt.xlabel('Iterations count')
plt.ylabel('PLSA perp. (blue)')
plt.grid(True)
#plt.show()

# ===================================================
# РАЗДЕЛ 2: Hегуляризация модели PLSA и новые метрики
# ===================================================

