# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 17:47:19 2020

@author: mb207
"""

import numpy as np
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt

data2 = pd.read_csv('./cmc.data',header=None)
data2.columns = ['age','wife_edu','husband_edu','child_num','religion','worked','husband_job','living_standard','media','contraception']
# data = np.loadcsv('C:/Users/mb207/Downloads/covtype.data')
np_arr = np.array(data2)
'''
前置處理開始
'''
Husbandoccupation_1 = np.where(np_arr[:,6] == 1,1,0)#對丈夫的工作做ont-hot_encoding
Husbandoccupation_2 = np.where(np_arr[:,6] == 2,1,0)
Husbandoccupation_3 = np.where(np_arr[:,6] == 3,1,0)
Husbandoccupation_4 = np.where(np_arr[:,6] == 4,1,0)
np_arr = np.column_stack((np_arr,Husbandoccupation_1))
np_arr = np.column_stack((np_arr,Husbandoccupation_2))
np_arr = np.column_stack((np_arr,Husbandoccupation_3))
np_arr = np.column_stack((np_arr,Husbandoccupation_4))
np_arr = np.delete(np_arr,6,1)#刪除原本丈夫的工作那一欄
'''
前置處理結束
'''
np.random.shuffle(np_arr)#打亂數據
data = np.column_stack((np_arr[:,:8],np_arr[:,9:]))#取data

label = np_arr[:,-5]#取label
data_80p = round(len(np_arr)*8/10)
train_data = data[:data_80p]#80%當資料
train_label = label[:data_80p]#80的label
test_data = data[data_80p:]#剩下的20%當作test
test_label = label[data_80p:]#20%的label
clf = tree.DecisionTreeClassifier(criterion = 'gini',max_depth = 5)  # ,splitter='best', min_samples_leaf = 20

clf = clf.fit(train_data, train_label)
fig, ax = plt.subplots(figsize=(12, 12))
tree.plot_tree(clf,fontsize=6) 

# print(sum(clf.predict(test_data)==test_label)/len(test_label))#算test的準確率

# print('Accuracy of the train data:'.format(clf.score(train_data, train_label)))
# print('Accuracy of the test data:'.format(clf.score(test_data, test_label)))
print(clf.score(train_data, train_label))
print(clf.score(test_data, test_label))

