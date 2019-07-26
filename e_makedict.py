import pickle
import numpy as np
from pathlib import Path
import json

def make_dict(path, pkl, enc):
    '''
    Creates dictionary that maps
    image URLs to their associated
    50-dimensional embeddings.

    Parameters:
    -----------
    path: Valid file PATH
    A path to a json containing metadata
    for COCO images, along with captions.
    pkl: Valid pickle file name
    A pickle file containing a dictionary
    that maps COCO images to the
    512-dimensional images.
    enc: Valid pickle file name
    A pickle file of a class containing
    an autoencoder.

    Returns:
    --------
    Dictionary{string, numpy.ndarray[float]}
    (size of arrays: (50,))
    A dictionary of image URLs mapping to
    50-dimensional embeddings.
    (This dictionary is also saved into
    a Pickle file.)
    '''

    res = {}

    with open(path) as json_file:
        data = json.load(json_file)
        print(data['images'][0])
    with open(pkl, "rb") as pickle_file:
        feat = pickle.load(pickle_file)
    with open(enc, "rb") as pick_file2:
        model = pickle.load(pick_file2)
    

    print(len(feat.keys()))

    for img in data['images']:
        curl = img['coco_url']
        if(img['id'] not in feat):
            continue
        vec = model(feat[img['id']])
        res[curl] = vec
    
    pickle_out = open("database.pickle", "wb")
    pickle.dump(res, pickle_out)
    pickle_out.close()

    return res