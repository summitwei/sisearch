import json,pickle
with open("captions_train2014.json") as file1:
    bigDic=json.load(file1)
with open("resnet18_features.pkl") as file2:
    smallDic=pickle,