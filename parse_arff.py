# coding: utf-8
"""Класс парсинга корпусов формата ARFF

   Скрипт рассчитан на работу с корпусами OHSUMED и RCV

"""
__author__ = 'Nikolay Karpov'

import pyparsing as p
import os
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier as mc
from scipy.sparse import csr_matrix

import codecs
import sys

import pickle
from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import SGDClassifier as SGDC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.preprocessing import MultiLabelBinarizer as mb
from sklearn import metrics


class Parse_ARFF:
    def __init__(self):
        pass

    def read_arff(self, _fname):
        """Парсит файл arff

           Сохраняет данные в особую структуру - см. ниже.

        Args:
            _fname: Путь к файлу данных.

        Returns:
            tokens.arffdata: Это по сути некая структура, в которой содержаться
                             tokens.arffdata.dataList - список из списков-докуметов разреженной матрицы arff,
                                                        каждый из которых является списком из пар:
                                                               - индекс термина (класса)
                                                               - значение термина (1.0 для класса)
                             tokens.arffdata.identifiers - простой список всех терминов и классов (словарь) arff

        """
        print "Parsing ARFF...", _fname
        text = ''.join(open(_fname, 'r').readlines())
        relationToken = p.Keyword('@RELATION', caseless=True)
        dataToken = p.Keyword('@DATA', caseless=True)
        attribToken = p.Keyword('@ATTRIBUTE', caseless=True)
        ident = p.ZeroOrMore(p.Suppress('\''))+p.Word(p.alphas, p.alphanums + '_-.').setName('identifier')+p.ZeroOrMore(p.Suppress('\''))
        relation = p.Suppress(relationToken) + p.ZeroOrMore(p.Suppress('"'))\
                   +ident.setResultsName('relation') + p.ZeroOrMore(p.Suppress('"'))
        attribute = p.Suppress(attribToken)+p.quotedString.setParseAction(lambda t: t.asList()[0].strip("'")).setResultsName('attrname') + p.Suppress(p.restOfLine)
        int_num = p.Word(p.nums)
        pm_sign = p.Optional(p.Suppress("+") | p.Literal("-"))
        float_num = p.Combine(pm_sign + int_num + p.Optional('.' + int_num) + p.Optional('e' + pm_sign + int_num)).setParseAction(lambda t: float(t.asList()[0]))
        module_name=p.Group((int_num.setParseAction(lambda t: int(t.asList()[0]))).setName('Key')+
                            (p.quotedString.setParseAction(lambda t: t.asList()[0].strip("'"))|float_num).setName('Value')+
                             p.Suppress(','))
        dataList = (p.Suppress('{')+p.OneOrMore(module_name)+p.Suppress('}')).setParseAction(lambda t: [t.asList()])
        comment = '%' + p.restOfLine
        arffFormat = (p.OneOrMore(p.Suppress(comment))+relation.setResultsName('relation') +
                       p.OneOrMore(attribute).setResultsName('identifiers')+
                       dataToken+
                       p.OneOrMore(dataList).setResultsName('dataList')
                      ).setResultsName('arffdata')
        tokens =  arffFormat.parseString(text)
        featureNames=tokens.arffdata.identifiers
        return (tokens.arffdata)

    # TODO убрать последние 2 агрумента
    def make_csr(self, _input, _feature_num=11286, _class_num=88):
        """Переводит объект данных arff в семейство разреженных матриц и список имен классов

        Args:
            _input: объект данных arff
            _feature_num: число терминов. QuantOHSUMED 11286 QuantRCV1 21610
            _class_num:   число классов.  QuantOHSUMED    88 QuantRCV1    99

        Returns:
            [csr, y, y_names]: Возвращает список, где
                               csr - разреженая матрица TFIDF (документ-термин)
                               y -   разреженная бинарная матрица с индексами классов (документ-класс)
                               y_names - имена классов
        """

        _indptr = [0]    # массив индексов документов матрицы документ-термин
        _indices = []    # массив индексов терминов матрицы документ-термин
        _data = []       # массив значений tfidf терминов матрицы документ-термин
        _data_names=[]   # массив алиасов документов, e.g. '88_54853'
        _classes_bin=[]  # бинарная (0/1) матрица документ-класс как список списков одной длины
        _i=0             # итератор документов
        _a0=[0 for i in range(_class_num)]

        for _element in _input.dataList:            # цикл по документам
            _class=[]                               # инициализируем массив индексов классов
            for _pair in _element:                  # цикл по парам аттрибутов
                if _pair[0] == 0:                   # атрибут, содержащий алиас документа, e.g. '88_54853'
                    _data_names.append(_pair[1])    # добавляем алиас в массив алиасов
                    __name=_pair[1]
                elif _pair[0]>_feature_num:         # атрибут принадлежности классу, e.g. 11360 1
                    _class.append(_pair[0])         # добавляем номер метки класса (11360) в массив индексов классов
                else:                               # атрибут термина
                    _indices.append(_pair[0]-1)     # добавляем индекс термина в массив индексов
                    _data.append(_pair[1])          # добавляем TFIDF термина в массив tfidf
                    _i += 1
            _indptr.append(_i)                      # в массив числовой сквозной нумерации документов

            #make matrix like _classes_bin=mb().fit_transform(_classes)

            _line=[0 for i in range(_class_num)]    # создаем вектор размером с число классов в коллекции
            for _it in _class:                      # цикл по индексам классов в текущем документе
                _line[_it-_feature_num-1] = 1       # вектор говорит, что текущий док принадлежит таким-то классам
            _classes_bin.append(_line)              # добавляем бинарный вектор в бинарную матрицу документ-класс

        _ident=_input.identifiers[_feature_num+1:]  # список меток классов

        # создаем объект разреженной матрицы размером <число документов x число терминов>
        _scr=csr_matrix((_data, _indices, _indptr),shape=[len(_classes_bin), _feature_num], dtype=float)

        return [_scr, csr_matrix(_classes_bin), _ident]

    def make_uci(self, _input, _feature_num=11286, _class_num=88):
        """Сохраняет объект данных arff в формате UCI Bag-of-words
           https://archive.ics.uci.edu/ml/datasets/Bag+of+Words

           Сделан на основе метода self.make_csr() этого же класса

        Args:
            _input: объект данных arff
            _feature_num: число терминов. QuantOHSUMED 11286 QuantRCV1 21610
            _class_num:   число классов.  QuantOHSUMED    88 QuantRCV1    99

        Returns: boolean

        """

        #_indptr = [0]    # массив индексов документов матрицы документ-термин
        #_indices = []    # массив индексов терминов матрицы документ-термин
        #tfidfs = []       # массив значений tfidf терминов матрицы документ-термин
        #doc_aliases=[]   # массив алиасов документов, e.g. '88_54853'
        #_classes_bin=[]  # бинарная (0/1) матрица документ-класс как список списков одной длины
        _i=0             # счетчик документов
        nnz = 0          # счетчик ненулевых элементов NNZ
        #_a0=[0 for i in range(_class_num)]

        collection_name = 'ohsumed'                                # название коллекции
        train_data_folder = os.path.join('train',collection_name)  # папка с обучающей выборкой

        # создать папку для UCI данных, если её нет
        if not os.path.exists(train_data_folder):
            os.makedirs(train_data_folder)

        docword_file_path = os.path.join(train_data_folder,'docword.' + collection_name + '.txt')
        vocab_file_path = os.path.join(train_data_folder,'vocab.' + collection_name + '.txt')

        # на мой взгляд, без этого цикла по документам NNZ никак не узнать на данном этапе
        for document in _input.dataList:
            nnz += len(document) - 1

        with codecs.open(docword_file_path, 'w', encoding='utf-8') as f:

            print >> f, len(_input.dataList)     # пишем в файл число документов
            print >> f, len(_input.identifiers)  # пишем в файл размер словаря (с учетом классов)
            print >> f, nnz                      # пишем NNZ

            for document in _input.dataList:            # цикл по документам
                _i += 1                                 # инкрементируем счетчик документов
                #_class=[]                               # инициализируем массив индексов классов
                for _pair in document:                  # цикл по парам аттрибутов
                    if _pair[0] == 0:                   # атрибут, содержащий алиас документа, e.g. '88_54853'
                        #doc_aliases.append(_pair[1])    # добавляем алиас в массив алиасов
                        # __name=_pair[1]
                        continue
                    elif _pair[0]>_feature_num:         # атрибут принадлежности классу, e.g. 11360 1
                    #    _class.append(_pair[0])         # добавляем номер метки класса (11360) в массив индексов классов
                         _pair[1] = int(_pair[1])        # принадлежность к классу дб целой единицей
                    #else:                               # атрибут термина
                    #    _indices.append(_pair[0]-1)     # добавляем индекс термина в массив индексов
                    #    tfidfs.append(_pair[1])          # добавляем TFIDF термина в массив tfidf

                    nnz += 1
                    print >> f, _i, _pair[0], _pair[1]   # пишем тройки

                #_indptr.append(_i)                      # в массив числовой сквозной нумерации документов

                #make matrix like _classes_bin=mb().fit_transform(_classes)
                #_line=[0 for i in range(_class_num)]    # создаем вектор размером с число классов в коллекции
                #for _it in _class:                      # цикл по индексам классов в текущем документе
                #    _line[_it-_feature_num-1] = 1       # вектор говорит, что текущий док принадлежит таким-то классам
                #_classes_bin.append(_line)              # добавляем бинарный вектор в бинарную матрицу документ-класс

            f.close()

        #_ident=_input.identifiers[_feature_num+1:]  # список меток классов

        with codecs.open(vocab_file_path, 'w', encoding='utf-8') as f:

            for term in _input.identifiers[1:_feature_num+1]:
                print >> f, term                               # пишем названия терминов
            for _class in _input.identifiers[_feature_num+1:]:
                print >> f, _class,'@class'                    # пишем названия классов со спецметкой для BigARTM

            f.close()

        # создаем объект разреженной матрицы размером <число документов x число терминов>
        #_scr=csr_matrix((_data, _indices, _indptr),shape=[len(_classes_bin), _feature_num], dtype=float)

        return True


    # TODO кажется не используется - удалить
    def make_binary(self, _input_y, _num=0):
        _y=[]
        _i=0
        for _line in _input_y:
            if _line[_num]==1:
                _y.append(1)
                _i += 1
            else:
                _y.append(-1)
        return _y

    # TODO кажется не используется - удалить
    def make_dat_file(self, _input_X, _y):
        _i=0
        for _index in _y:
            _str='%d' %_index
            #for _doc in _input_X.getrow(_i):
                #print(_doc)
            #print(_str)
            _i=_i+1
        return 0

    def read_dir(self, _path):
        """Парсит папку с файлами данных, и разделяет пути на 2 массива обучающих и тестовых
         в зависимости от наличия в имени файла подстрок train или test

        Args:
            _path: Путь к папке с файлами данных.

        Returns:
            (_file_train,_files_test): Возвращает списки путей к файлам данных

        """
        _files_test=[]
        _file_train=''
        _list_dir=os.listdir(_path)
        for _file in _list_dir:                     # цикл по файлам в директории
            file_path = os.path.join(_path,_file)
            if _file.find('train') > 0:             # если имя файла содержит "train"
                _file_train = file_path             # добавляем путь к файлу в обучающий массив
            elif _file.find('test') > 0:            # если имя файла содержит "test"
                _files_test.append(file_path)       # добавляем путь к файлу в тестовый массив
        #print(_file_train, _files_test)
        return _file_train, _files_test

    # TODO кажется не используется - удалить
    def fit(self, _input_X, _input_y):
        #classif=LR( )
        classif = mc(SVC(kernel='linear', probability=True))#rbf poly sigmoid
        model=classif.fit(_input_X, _input_y)
        return model

    def convert_arff(self, dir_name='QuantOHSUMED', is_predict=False):
            # Read ARFF files;
            # TODO serialize data to pickle format; - это убрать
            # learn ML model predict probabilities and serialize results to pickle format

        if dir_name=='QuantOHSUMED':# QuantOHSUMED num_of_feat=11286 num_of_classes=88
            num_of_feat=11286
            num_of_classes=88
        elif dir_name=='QuantRCV1':# QuantRCV1 num_of_feat=21610 num_of_classes=99
            num_of_feat=21610
            num_of_classes=99

        train_file, test_files = self.read_dir(dir_name)  # получаем списки путей к файлам данных
        arff=self.read_arff(train_file)                   # получаем токены обучающей выборки

        [csr, y, y_names]=self.make_csr(arff, num_of_feat, num_of_classes)

        # делаем дамп данных обучающей выборки в pickle
        #with open('texts/pickle_'+train_file+'.pickle', 'wb') as f:
        #    pickle.dump([csr, y, y_names], f, protocol=2)
        #    f.close()

        #if is_predict:
        #    model=self.fit(csr,y)
        prob_list=[]
        y1_list=[]

        for test_file in test_files:  # цикл по файлам тестовой выборки
            arff1=self.read_arff(test_file) # получаем токены

            [csr1, y1, y1_names]=self.make_csr(arff1, num_of_feat, num_of_classes)

            #with open('texts/pickle_'+test_file+'.pickle', 'wb') as f:
            #    pickle.dump([csr1, y1, y1_names], f, protocol=2)
            #    f.close()
            #print('texts/pickle_'+test_file+'.pickle')

            #if is_predict:
            #    prob_y1 = model.predict_proba(csr1)
            #    print(metrics.classification_report(y1,pr_y1))
            #    prob_list.append(prob_y1)
            #    y1_list.append(y1)
            #    with open('texts/cl_prob_'+test_file+'.cl_prob', 'wb') as f:
            #        pickle.dump(prob_y1, f, protocol=2)
            #        f.close()

        #if is_predict:
        #    with open('texts/cl_prob_'+dir_name+'.pickle', 'wb') as f:
        #        pickle.dump([y, y1_list, prob_list, test_files, y_names], f, protocol=2)
        #        f.close()
        return 0
#pa=Parse_ARFF()
#pa.convert_arff('QuantOHSUMED') # 'QuantRCV1'