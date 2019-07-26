import json,pickle,random,sys,time,string

import mygrad as mg
# import mynn
import numpy as np
import gensim
from gensim.models.keyedvectors import KeyedVectors
path = r"./glove.6B.50d.txt.w2v"
t0 = time.time()
glove = KeyedVectors.load_word2vec_format(path, binary=False)
t1 = time.time()
print("elapsed %ss" % (t1 - t0))
# glove = KeyedVectors.load_word2vec_format(r"./glove.6B.50d.txt.w2v", binary=False)
from mynn.initializers.he_normal import he_normal
from mynn.layers.dense import dense
# from mynn.losses.mean_squared_loss import mean_squared_loss
# from mynn.optimizers.sgd import SGD
from mynn.optimizers.adam import Adam
# import matplotlib.pyplot as plt
# from noggin import create_plot
# plotter, fig, ax = create_plot(metrics=["loss"], max_fraction_spent_plotting=.75,last_n_batches=5000)
with open("captions_train2014.json") as file1:
    bigDict=json.load(file1)
with open("resnet18_features.pkl","rb") as file2:
    smallDict=pickle.load(file2)
index=0
idToCaption={}
for cc in bigDict["annotations"]:
    # print(cc['image_id'])
    caption2=cc['caption'][0:].lower()
    caption=""
    for bb in caption2:
        if bb.isalpha() or bb is " ":
            caption+=bb
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
    if cc['image_id'] in idToCaption:
        idToCaption[cc["image_id"]].add(tuple(captionEmbed))
    else:
        idToCaption[cc["image_id"]]=set()
        idToCaption[cc["image_id"]].add(tuple(captionEmbed))
id2=0
temp=set(smallDict.keys())
temp2=set(idToCaption.keys())
for key in temp2:
    if id2%2000==0:
        print(id2)
    id2+=1
    if key not in temp:
        del idToCaption[key]

# # print(idToCaption[384029])
# for a in bigDict['images']:
#     print((idToCaption[a['id']],smallDict[a['id']].shape,smallDict[bigDict['images'][random.randint(0,len(bigDict['images']))]['id']].shape))
#     index+=1
#     if index>5:
#         sys.exit(0)


class LinearAutoencoder:
    def __init__(self, D_full, D_out):

        self.dense1 = dense(D_full, D_out, weight_initializer=he_normal, bias=True)

    def __call__(self, x):
        '''Passes data as input to our model, performing a "forward-pass".

        This allows us to conveniently initialize a model `m` and then send data through it
        to be classified by calling `m(x)`.

        Parameters
        ----------
        x : Union[numpy.ndarray, mygrad.Tensor], shape=(M, D_full)
            A batch of data consisting of M pieces of data,
            each with a dimentionality of D_full.

        Returns
        -------
        mygrad.Tensor, shape=(M, D_full)
            The model's prediction for each of the M pieces of data.
        '''
        return self.dense1(x)  # <COGLINE>

    @property
    def parameters(self):
        """ A convenience function for getting all the parameters of our model.

        This can be accessed as an attribute, via `model.parameters`

        Returns
        -------
        Tuple[Tensor, ...]
            A tuple containing all of the learnable parameters for our model """
        return self.dense1.parameters
model=LinearAutoencoder(512,50)
optim=Adam(model.parameters)
useful=set(idToCaption.keys())
useful2=set(smallDict.keys())
lossSoFa=[]
index=0
while  index<5000000:
    if index%100000==0:
        ans=input("Do you want to continue: 1 or 0")
        if int(ans)==0:
            pickle.dump(model, open("model.pkl", "wb"))
            sys.exit(0)
            # Pickle dump
    goodFeature=[]
    badFeature=[]
    captionEmbed=[]
    for nfl in range(500):
        id=0
        while id not in useful:
            id=random.sample(bigDict["images"],1)[0]["id"]
        captionEmbed.append(np.array(random.sample(idToCaption[id],1)[0]))
    # caption=caption.translate(None, string.punctuation).lower()
        goodFeature.append(model(smallDict[id]))

    # print(caption)
        badID=0
        while badID not in useful:
            badID=random.sample(bigDict["images"],1)[0]["id"]
        badFeature.append(model(smallDict[badID]))


    loss=mg.nnet.margin_ranking_loss(goodFeature@captionEmbed,badFeature@captionEmbed,1,0.1)
    lossSoFa.append(loss.item())
    loss.backward()
    optim.step()
    loss.null_gradients()
    # plotter.set_train_batch({"loss": loss.item()}, batch_size=500)
    if index%2000==0:
    #     plotter.set_train_epoch()
        print("a:%s, loss:%s"%(index,np.mean(lossSoFa)))
        # lossSoFa=[]
    # loss=mg.nnet.margin_ranking_loss()
    index+=1
    if np.mean(lossSoFa)<.0001 and np.mean(lossSoFa)>0:
        pickle.dump(model, open("model.pkl", "wb"))
        sys.exit(2)
pickle.dump(model,open("model.pkl","wb"))



