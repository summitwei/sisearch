import gensim
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import gensim

glove = KeyedVectors.load_word2vec_format(r"./glove.6B.50d.txt.w2v", binary=False)
from urllib.request import urlopen
import matplotlib.pyplot as plt

def se_text(text):
    '''This embeds a string by turning it into a list then summing the glove values. The out put is the normalized output'''
    l = text.split()     # makes a list for all the words
    ret= np.zeros((1,50))  #Size of vectors
    for i in l:
        ret+= glove[i]   #Combines all vectors
    return ret/len(l) #Normalizes
    #Do we have repeates

def se_image(path, url):#path is the pickle file
    ''' This takes in the url and returns the np.array'''
    # downloading the data


    # converting the downloaded bytes into a numpy-array
    img = plt.imread(data, format='jpg')  # shape-(460, 640, 3)

    with open(path, "rb") as f:
        data = urlopen(url)

    # displaying the image
    fig, ax = plt.subplots()
    ax.imshow(img)

def Sim(text, pice ):
    '''give the semantic text, and semantic picture'''
    return np.dot(text,pice)
def find(text, hyp):
    pic = data
    l=[]
    v = Sim(text,pic)
    if v >hyp:
        l.append((pic, v))
    if l is not None:
        y = lambda item : item[1]
        l.sorted(key=y)
        return l[0]
#print(se_text("hello world").shape)
#path="C:\Users\Charles Richards\Desktop\PortableGit\sisearch\resnet18_features.pkl"