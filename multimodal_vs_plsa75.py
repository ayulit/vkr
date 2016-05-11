# -*- coding: utf-8 -*-
"""Сравнение мультимодальной модели со стандартным тематическим моделированием PLSA

   Строится график PLSA perp. (blue), ARTM perp. (red)

   Requirements: BigArtm v 0.7.5

   Работает с коллекциями BANKS и OHUSMED

"""

import artm
import os
import glob
import matplotlib.pyplot as plt

collection_name = 'ohsumed'
train_data_folder = os.path.join('train', collection_name)  # папка с обучающей выборкой

target_folder = os.path.join('target', collection_name)
target_train = os.path.join(target_folder, 'train')

dictionary_name = 'dictionary'
dictionary_filename = dictionary_name + '.dict'
dictionary_path = os.path.join(target_train, dictionary_filename)

dictionary_filename_txt = dictionary_name + '.txt'
dictionary_path_txt = os.path.join(target_train, dictionary_filename_txt)

#_topics_count = 3
#_tau_phi = -0.1
#_tau_decor = 1.5e+5
#_tau_theta = -0.15

batch_vectorizer = None  # инициализция ссылки на BatchVectorizer
batches_found = len(glob.glob(os.path.join(target_folder, '*.batch')))
if batches_found < 1:
    print "No batches found, parsing them from textual collection...",
    # создаем из UCI батчи и сохраняем их в traget folder
    batch_vectorizer = artm.BatchVectorizer(data_path=train_data_folder,
                                            data_format='bow_uci',
                                            collection_name=collection_name,
                                            target_folder=target_train)
    print " OK."
else:
    print "Found " + str(batches_found) + " batches, using them."
    # создаем из UCI батчи и сохраняем их в traget folder
    batch_vectorizer = artm.BatchVectorizer(data_path=target_train, data_format='batches')

# теперь разберемся со словарем

# как ни уверяют разработчики в туториале, создать просто так объект Dictionary тут не получится,
# поэтому перед созданием этого объекта зачем то нужно создавать хоть какую, но модель
model_for_dic = artm.ARTM()

# и только потом создаем словарь, если его нет
# vocab_file подсовываем для соблюдения порядка в словаре bigartm
if not os.path.isfile(dictionary_path):
    # собираем словарь
    model_for_dic.gather_dictionary(dictionary_target_name=dictionary_name,
                                    data_path=target_train,
                                    vocab_file_path=os.path.join(train_data_folder, 'vocab.' + collection_name + '.txt'))
    # сохраняем словарь в бинарном виде (автоматом подставляется расширение .dict)
    model_for_dic.save_dictionary(dictionary_name=dictionary_name, dictionary_path=dictionary_path)
    # сохраняем словарь в текстовом виде
    model_for_dic.save_text_dictionary(dictionary_name=dictionary_name, dictionary_path=dictionary_path_txt)


# создаем модель с весами модальности в конструкторе
# метки классов в 10 раз более влиятельны, чем обычные слова
model_artm = artm.ARTM(num_topics=20, class_ids={'@default_class': 1.0, '@labels': 10.0})

# создаем модель plsa
model_plsa = artm.ARTM(num_topics=20)

# вот это получается и не надо, т.к. в общем случае словарь не нужен, а нужны только батчи
# словарь нужен для метрик походу дела!
# но оказалось, что без этого тоже никак
model_plsa.load_dictionary(dictionary_name=dictionary_name, dictionary_path=dictionary_path)
model_artm.load_dictionary(dictionary_name=dictionary_name, dictionary_path=dictionary_path)
model_plsa.initialize(dictionary_name=dictionary_name)
model_artm.initialize(dictionary_name=dictionary_name)

# Подключаем модели ARTM метрику перплексии (метрика - как объект) для все модальностей
# Для class_ids отсутствие значения означает использование всех имеющихся в модели модальностей!
model_artm.scores.add(artm.PerplexityScore(name='perplexity_score_artm',
                                      use_unigram_document_model=False,
                                      #class_ids=['@default_class'],
                                      dictionary_name=dictionary_name))

# Подключаем модели PLSA метрику перплексии (метрика - как объект)
model_plsa.scores.add(artm.PerplexityScore(name='perplexity_score_plsa',
                                      use_unigram_document_model=False,
                                      dictionary_name=dictionary_name))

# Configure basic regularizers



# добавим в модель метрику топа токенов для каждой из модальностей
#model_artm.scores.add(artm.TopTokensScore(name='top_tokens_score_def', num_tokens = 6,  class_id='@default_class'))
#model_artm.scores.add(artm.TopTokensScore(name='top_tokens_score_lab', num_tokens = 6,  class_id='@labels'))

# добавим в модель метрику разреженность Φ для каждой из модальностей
#model_artm.scores.add(artm.SparsityPhiScore(name='sparsity_phi_score_def', class_id='@default_class'))
#model_artm.scores.add(artm.SparsityPhiScore(name='sparsity_phi_score_lab', class_id='@labels'))

# в SparsityThetaScore нельзя добавить инфу о классах!
#model_artm.scores.add(artm.SparsityThetaScore(name='SparsityThetaScore'))

# Обучим модели, установив 1 проходов по документу и сделав еще 9 итераций по коллекции
model_artm.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=9, num_document_passes=1)
model_plsa.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=9, num_document_passes=1)

if False:

# зырим на метрики

    print 'sparsity_phi_score_def'
    print model_artm.score_tracker['sparsity_phi_score_def'].value    # .last_value
    print 'top_tokens_score_lab'
    print model_artm.score_tracker['sparsity_phi_score_lab'].value    # .last_value
    # Retrieve and visualize top tokens in each topic
    saved_top_tokens_def = model_artm.score_tracker['top_tokens_score_def'].last_topic_info
    print 'top_tokens_score_def'
    for topic_name in model_artm.topic_names:
        print topic_name + ': ',
        print saved_top_tokens_def[topic_name].tokens
    saved_top_tokens_lab = model_artm.score_tracker['top_tokens_score_lab'].last_topic_info
    print 'top_tokens_score_lab'
    for topic_name in model_artm.topic_names:
        print topic_name + ': ',
        print saved_top_tokens_lab[topic_name].tokens


# Динамика значений перплексии за проходы
print 'perplexity_score_artm'
print model_artm.score_tracker['perplexity_score_artm'].value # .last_value
print 'perplexity_score_plsa'
print model_plsa.score_tracker['perplexity_score_plsa'].value # .last_value

# График динамики перплексий
plt.plot(xrange(model_plsa.num_phi_updates), model_plsa.score_tracker['perplexity_score_plsa'].value, 'b--',
         xrange(model_artm.num_phi_updates), model_artm.score_tracker['perplexity_score_artm'].value, 'r--', linewidth=2)
plt.xlabel('Iterations count')
plt.ylabel('PLSA perp. (blue), ARTM perp. (red)')
plt.grid(True)
plt.show()
