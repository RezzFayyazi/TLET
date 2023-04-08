import torch
from fastai import *
from fastai.text import *
from fastai.vision.all import *
from fastai.text.all import *
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
import numpy as np
from torch.cuda import is_available
import torch
from ExploitTypePredictor import ExploitTypesPredictor

ft_cls_exploitdb_path = 'exploitdb.csv'
df_cls_exploitdb = pd.read_csv(ft_cls_exploitdb_path)
df_cls_exploitdb = shuffle(df_cls_exploitdb)

df_exploit_descriptions = df_cls_exploitdb.iloc[:,2:3]
df_exploit_labels = df_cls_exploitdb.iloc[:,5:6]
df_exploit_labels.replace(['dos', 'local', 'remote', 'webapps'], [0,1,2,3],inplace=True)

pred_model = ExploitTypesPredictor()
predictions = []

_, test_x, _, test_y = train_test_split(df_exploit_descriptions, df_exploit_labels, test_size = 0.2, random_state=2022)

test_descs = test_x.iloc[:,0:1].values
for i in range(len(test_descs)):
    pred = pred_model.predict(test_descs[i,0])
    predictions.append(pred[1])
print(predictions)
true_labels = test_y
print(true_labels)
rep = classification_report(true_labels, predictions, target_names =['dos', 'local', 'remote', 'webapps'])
print(rep)