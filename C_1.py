import gensim
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import gensim
import pickle
from collections import Counter
glove = KeyedVectors.load_word2vec_format(r"./glove.6B.50d.txt.w2v", binary=False)
from urllib.request import urlopen
import matplotlib.pyplot as plt
i = lambda f, TotalWords: np.log(f/TotalWords)
TotalWords =0
database = r"C:\Users\Charles Richards\Desktop\PortableGit\sisearch\database.pickle"
with open(database,"rb") as file:
    database=pickle.load(file)
def se_text(text):
    # '''This embeds a string by turning it into a list then summing the glove values. The out put is the normalized output'''
#     # l = text.split()     # makes a list for all the words
#     # ret= np.zeros((1,50))  #Size of vectors
#     # TotalWords = len(l)
#     # f = dict(Counter(l))
#     #
#     #
#     # for gg in l:
#     #     idf = i(f[gg],TotalWords)
#     #     ret+= idf*glove[gg]   #Combines all vectors
#     #
#     #
#     #
#     # return ret/np.linalg.norm(ret) #Normalizes
    caption2 = text.lower()
    caption = ""
    for bb in caption2:
        if bb.isalpha() or bb is " ":
            caption += bb
    try:
        captionEmbed = np.array(glove[caption.split()[0]])
    except:
        captionEmbed = np.zeros((50,))
    removed = 0
    for g in caption.split()[1:]:
        try:
            captionEmbed += np.array(glove[g])
        except:
            # print(captionEmbed.shape)
            captionEmbed += np.zeros((50,))
            removed += 1
    captionEmbed = captionEmbed / (len(caption.split()) - removed)
    return captionEmbed


#Do we have repeates
#data = database[url]
def se_image(url):#path is the pickle file
    ''' This takes in the url and returns the np.array'''
    # downloading the data


    # converting the downloaded bytes into a numpy-array

     # shape-(460, 640, 3)

    # with open(path, "rb") as f:
    print(url)
    data = urlopen(url[0])
    # data=data.flatten()

    img = plt.imread(data, format='jpg')
    # displaying the image
    fig, ax = plt.subplots()
    ax.imshow(img)
    plt.show()

def Sim(text, pice ):
    '''give the semantic text, and semantic picture'''
    return np.dot(text,pice.reshape(50,1))
def find(text,  hyp=.75):
    # pic = data
    l=[]

    pickle = database
    o = se_text(text)

    for pics in pickle:
        v = Sim(o, pickle[pics])
        # print(v)
        # print(hyp)
        if v[0] >hyp:
            l.append((pics, v))
        if l is not None:
            y = lambda item : item[1]
    l.sort(key=y,reverse=True)
    if len(l)>0:
        se_image(l[0])
    else:
        print("Image match not found")
        return


#print(se_text("hello world").shape)
#Sum(idf(word)*that word's embedding) all of the words not for a chalor but for a np.array(50,)

text=input("Enter string to search").lower()
find(text,hyp=.75)