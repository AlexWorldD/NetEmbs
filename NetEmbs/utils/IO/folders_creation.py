# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
create_work_folder.py
Created by lex at 2019-07-04.
"""
import os
from typing import Optional


def create_working_folder(path: Optional[str] = None) -> None:
    """
    Helper function to construct folders structure for the current run

    For each execution creates the following hierarchy of folders w.r.t. the parameters:
                                                                                    <Sampling>/<Skip-Gram>/<TensorFlow>
    For each execution finWalk caches
        - sampling sequences,
        - Skip-Grams
        - obtained embeddings
    and stores it in an appropriate folder.

    Parameters
    ----------
    path : str, default: None
            String name to folder for the current execution.
            If None, use information from CONFIG

    Returns
    -------
        None
    """
    if path is None:
        from NetEmbs import CONFIG
        path = CONFIG.WORK_FOLDER[0] + CONFIG.WORK_FOLDER[1] + CONFIG.WORK_FOLDER[2]
    create_folder(path)
    create_folder(path + "img/")
    print("Working directory is ", path)


def create_folder(path: str) -> None:
    """
    Helper function to safety create folder
    Parameters
    ----------
    path : Path to folder to be created

    Returns
    -------
        None
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
