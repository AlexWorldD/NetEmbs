# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
create_work_folder.py
Created by lex at 2019-07-04.
"""
import os
from NetEmbs import CONFIG


def create_working_folder():
    # Create working folder for current execution
    if not os.path.exists(CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + CONFIG.WORK_FOLDER[2]):
        os.makedirs(CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + CONFIG.WORK_FOLDER[2], exist_ok=True)
    if not os.path.exists(CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + CONFIG.WORK_FOLDER[2] + "img/"):
        os.makedirs(CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + CONFIG.WORK_FOLDER[2] + "img/", exist_ok=True)
    print("Working directory is ", CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + CONFIG.WORK_FOLDER[2])


def create_folder(path):
    # Create working folder for current execution
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
