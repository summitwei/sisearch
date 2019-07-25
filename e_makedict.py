import pickle
import numpy as np
from pathlib import Path
import json

def make_dict(path):
    '''
    Creates dictionary that maps
    image URLs to their associated
    50-dimensional embeddings.

    Parameters:
    -----------
    path: Valid file PATH
    A path to a json containing metadata
    for COCO images, along with captions.

    Returns:
    --------
    Dictionary{string, numpy.ndarray[float]}
    (size of arrays: (50,))
    A dictionary of image URLs mapping to
    50-dimensional embeddings.
    (This dictionary is also saved into
    a Pickle file.)

    '''
    json_data = json.loads(r"./captions_train2014.py")[0]
    print(json_data['info'])