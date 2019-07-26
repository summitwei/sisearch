import gensim
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import gensim
from collections import Counter
glove = KeyedVectors.load_word2vec_format(r"./glove.6B.50d.txt.w2v", binary=False)
from urllib.request import urlopen
import matplotlib.pyplot as plt
i = lambda f, TotalWords: np.log(f/TotalWords)
TotalWords =0
database = "Does not exist"
def se_text(text):
    '''This embeds a string by turning it into a list then summing the glove values. The out put is the normalized output'''
    l = text.split()     # makes a list for all the words
    ret= np.zeros((1,50))  #Size of vectors
    TotalWords = len(l)
    f = dict(Counter(l))
    o = se_text(text)

    for i in l:
        idf = i(f(i))
        ret+= idf*glove[i]   #Combines all vectors



    return ret/np.linalg.norm(ret) #Normalizes
    #Do we have repeates
#data = database[url]
def se_image(path, url):#path is the pickle file
    ''' This takes in the url and returns the np.array'''
    # downloading the data


    # converting the downloaded bytes into a numpy-array

    img = plt.imread(data, format='jpg')  # shape-(460, 640, 3)

    with open(path, "rb") as f:
        data = urlopen(url)
    data=data.flatten()

    # displaying the image
    fig, ax = plt.subplots()
    ax.imshow(data)

def Sim(text, pice ):
    '''give the semantic text, and semantic picture'''
    return np.dot(text,pice)
def find(text, hyp=.75):
    pic = data
    l=[]

    pickle = database
    v = Sim(text,pic)
    for pics in pickle:
        for pic in pics:
            if v >hyp:
                l.append((pic, v))
            if l is not None:
                y = lambda item : item[1]
                l.sorted(key=y)
                return l[0]

#print(se_text("hello world").shape)
#Sum(idf(word)*that word's embedding) all of the words not for a chalor but for a np.array(50,)