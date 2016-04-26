# coding: utf-8
"""Проверка файла docword

   Скрипт чекает соответствие параметров

"""


import copy
import os
import sys

# =====================================================================================
# Описание функций
# =====================================================================================

def find_missing_items(int_list):

    # Finds missing integer within an unsorted list
    # and return a list of missing items
    # http://codereview.stackexchange.com/questions/24520/finding-missing-items-in-an-int-list/

    # Put the list in a set, find smallest and largest items
    original_set  = set(int_list)
    smallest_item = min(original_set)
    largest_item  = max(original_set)

    # Create a super set of all items from smallest to largest
    full_set = set(xrange(smallest_item, largest_item + 1))

    # Missing items are the ones that are in the full_set, but not in
    # the original_set
    return sorted(list(full_set - original_set))

# =====================================================================================
# Инициализация переменных
# =====================================================================================

collection_name = 'ohsumed'

train_data_folder = os.path.join('train', collection_name)
test_data_folder = 'test'

train_docword_file_path = os.path.join(train_data_folder, 'docword.' + collection_name + '.txt')
test_docword_file_path = os.path.join(test_data_folder, 'docword.' + collection_name + '.txt')


# =====================================================================================
# TRAIN
# =====================================================================================


if not os.path.exists(train_data_folder):
    print "Error: couldn't find TRAIN data. Skipping..."
else:
    lst = []
    with open(train_docword_file_path) as f:
        for line in f:
            lst.append(int(line.split()[0]))
        f.close()
    docs = copy.deepcopy(lst[0])  # число документов на входе
    # удаляем шапку
    for i in xrange(0, 3):
        # print "del", lst[0]
        del lst[0]
    doc_ids = list(set(lst))  # сет значений docID
    # Если число документов на входе не совпадает
    # с реальным количеством уникальных docID
    if len(doc_ids) != docs:
        print '\nWarning: TRAIN missing docIDs have found ',
        if (doc_ids[0] != 1) or (doc_ids[-1] != docs):
            print
        else:
            print find_missing_items(doc_ids)
    else:
        print 'TRAIN Ok.'

# =====================================================================================
# TEST
# =====================================================================================

if not os.path.exists(test_docword_file_path):
    print "Error: couldn't find TEST data. Skipping..."
else:
    lst = []
    with open(test_docword_file_path) as f:
        for line in f:
            lst.append(int(line.split()[0]))
        f.close()
    docs = copy.deepcopy(lst[0])  # число документов на входе
    # удаляем шапку
    for i in xrange(0, 3):
        # print "del", lst[0]
        del lst[0]
    doc_ids = list(set(lst))  # сет значений docID
    # Если число документов на входе не совпадает
    # с реальным количеством уникальных docID
    if len(doc_ids) != docs:
        print '\nWarning: TEST missing docIDs have found ',
        if (doc_ids[0] != 1) or (doc_ids[-1] != docs):
            print
        else:
            print find_missing_items(doc_ids)
    else:
        print 'TEST Ok.'

# print 'STOP'
# sys.exit()  # stop execution