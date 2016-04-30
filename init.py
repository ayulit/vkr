# coding: utf-8
"""Конфигурация среды

   Скрипт создает необходимые папки итп

"""

import os

target_folder_name = 'target'  # target папка для хранения данных BigARTM

# создать папку target, если её нет
if not os.path.exists(target_folder_name):
    os.makedirs(target_folder_name)