# -*- coding: utf-8 -*-
"""Тестим мультимодальность

   Данные захардкожены

"""

import string
import uuid

import artm.messages_pb2
import artm.library

body = []  # ex english
kwds = []  # ex russian

# в коллекции 7 доков, априорно можно увидеть, что общих тематики примерно 2: назовем их условно
# Topic#1      Topic#2
# Математика   Химия

#0 химия
body.append(u"Смесь спирт вода песок сахар медь бензин компонента разделение")
kwds.append(u"химия кулинария металлургия нефтеобработка")

#1 химия
body.append(u"Вещество ракетный двигатель тяга ускорения энергия химическая реакция")
kwds.append(u"rocketScience химия физика")

#2 химия
body.append(u"Наследственность живой организм ДНК полепептид размножение потомство")
kwds.append(u"биология химия генетика")

#3 математика
body.append(u"Инерциальая система отсчета ускорение материальный точка сила масса")
kwds.append(u"физика математика геометрия")

#4 математика
body.append(u"Задача кратчайший путь поиск между две точка граф минимум вес ребро язык Java")
kwds.append(u"математика информатика")

#5 математика
body.append(u"Распределение вероятность событие итерация алгоритм цикл процедура")
kwds.append(u"информатика статистика")

#6 математика
body.append(u"Количество ген итерация вероятность распределение")
kwds.append(u"биология информатика")

body_dic = {}  # mapping from body token to its index in batch.token list
kwds_dic = {}  # mapping from keywords token to its index in batch.token list
batch = artm.messages_pb2.Batch()  # batch representing the entire collection
batch.id = str(uuid.uuid1())
unique_tokens = artm.messages_pb2.DictionaryConfig()  # BigARTM dictionary to initialize model


def append(tokens, dic, item, class_id):
    """Парсит 2 соответствующие коллекции тела документа и его ключевых слов
       и заполняет её данными структуры BigArtm

    Args:
        tokens: список токенов (тела дока/ключевиков)
        dic: словарь (тела дока/ключевиков)
        item: объект документа BigArtm
        class_id: метка класса!

    Returns:
        void: Возвращает ничего

    """
    for token in tokens:
        if not dic.has_key(token):              # New token discovered: (т.е. если такой токен еще не добавлен)
            dic[token] = len(batch.token)       # 1. заполняем словари Python
            batch.token.append(token)           # 2. update batch.token and batch.class_id () (добавляем токен в батч)
            batch.class_id.append(class_id)
            entry = unique_tokens.entry.add()   # 3. update unique_tokens (заполняем словарь BigArtm)
            entry.key_token = token
            entry.class_id = class_id  # то есть метка класса соответствует токену!

        # Add token to the item.
        item.field[0].token_id.append(dic[token]) # сам токен
        # приращение частоты токена
        item.field[0].token_count.append(1)     # <- replace '1' with the actual number of token occupancies in the item


# Iterate through all items and populate the batch
for (bd, kw) in zip(body, kwds): # для каждой пары параллельных документов
    next_item = batch.item.add() # создаем документ в батче
    next_item.id = len(batch.item) - 1  #
    next_item.field.add() # добавляем поле в итем

    # Заполняем структуры BigArtm данными коллекции
    append(string.split(bd.lower()), body_dic, next_item, '@body')
    append(string.split(kw.lower()), kwds_dic, next_item, '@keywords')


# Create master component and infer topic model
with artm.library.MasterComponent() as master:
    unique_tokens.name = 'dictionary'  # имя словаря BigArtm
    dictionary = master.CreateDictionary(unique_tokens)  # создаем словарь BigArtm

    # ===================================================
    # Конфигурирование базовых показателей score
    # ===================================================

    # Create one top-token score per each class_id
    bd_top_tokens_score = master.CreateTopTokensScore(class_id='@body')  # топ токенов статьи
    kw_top_tokens_score = master.CreateTopTokensScore(class_id='@keywords')  # топ ключевиков

    bd_sparsity = master.CreateSparsityPhiScore(class_id='@body')
    kw_sparsity = master.CreateSparsityPhiScore(class_id='@keywords')

    theta_sparsity = master.CreateSparsityThetaScore()
    theta_snippet_score = master.CreateThetaSnippetScore()


    # Create and initialize model. Our expert knowledge says we need 2 topics ;)
    model = master.CreateModel(topics_count=2, inner_iterations_count=10,
                               class_ids=('@body', '@keywords'),
                               class_weights=(1.00, 1.00))
    model.Initialize('dictionary')  # Setup initial approximation for Phi matrix.

    # Infer the model in 10 passes over the batch
    for iteration in range(0, 10):
        master.AddBatch(batch=batch)
        master.WaitIdle()    # wait for all batches are processed
        model.Synchronize()  # synchronize model

    # Retrieve and visualize top tokens in each topic
    print 'body'
    artm.library.Visualizers.PrintTopTokensScore(bd_top_tokens_score.GetValue(model))
    print 'keywords'
    artm.library.Visualizers.PrintTopTokensScore(kw_top_tokens_score.GetValue(model))
    artm.library.Visualizers.PrintThetaSnippetScore(theta_snippet_score.GetValue(model))

    bd_phi, bd_numpy_matrix = master.GetTopicModel(model=model, class_ids={"@body"})
    kw_phi, kw_numpy_matrix = master.GetTopicModel(model=model, class_ids={"@keywords"})
    combined_phi, combined_numpy_matrix = master.GetTopicModel(model=model)

    print "\nSparsity of theta matrix = %.3f" % theta_sparsity.GetValue(model).value
    print "@body: phi matrix sparsity = %.3f," % bd_sparsity.GetValue(model).value, ' #tokens=%i' % len(bd_phi.token)
    print "@keywords: phi matrix sparsity = %.3f," % kw_sparsity.GetValue(model).value, ' #tokens=%i' % len(kw_phi.token)
