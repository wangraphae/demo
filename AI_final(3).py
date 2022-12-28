# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 19:44:23 2022

@author: danie
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

train = pd.read_csv('C:/Users/danie/Desktop/期末報告/train.csv')
test = pd.read_csv('C:/Users/danie/Desktop/期末報告/test.csv')

train_images = train.iloc[:, 1:].values.reshape(-1, 28, 28)

plt.imshow(train_images[35], cmap='gray') #看圖片
print(train.iloc[35, 0]) #此圖數字

X_train, X_val, y_train, y_val = train_test_split(train_images, train.iloc[:, 0], test_size=0.3, random_state=0)
#把訓練Data(圖片)與訓練答案分割 7成訓練 3成驗證
#train_data,test_data,train_target,test_target
print(X_train.shape)
print(X_val.shape)
#標準化

X_train  = StandardScaler().fit_transform(X_train .reshape(-1, 28*28)).reshape(-1, 28, 28)
#拉成1條做fit_transform，再還原成圖片樣子
X_val = StandardScaler().fit_transform(X_val.reshape(-1, 28*28)).reshape(-1, 28, 28)
#拉成1條做fit_transform，再還原成圖片樣子

plt.imshow(X_train[40], cmap='gray') #標準化完成後圖片

#預測機
#說明

"""隨機森林演算法會對資料從列方向（觀測值方向）
與欄方向（變數方向）進行 Bootstrap sampling，得到不同的訓練資料，，
然後根據這些訓練資料得到一系列的決策樹分類器，假如產生了 5 個決策樹分類器，
她們對某個觀測值的預測結果分別為 1, 0, 1, 1, 1，那麼隨機森林演算法的
輸出結果就會是 1"""
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train.reshape(-1, 28*28), y_train)
y_pred = rfc.predict(X_val.reshape(-1, 28*28))
print('Accuracy Score: ', accuracy_score(y_val, y_pred))

from sklearn import metrics
print(metrics.classification_report(y_val, y_pred, digits=2)) #比對預測是否精確
#F1值在1时达到最佳值（完美的精确度和召回率）
#https://www.cnblogs.com/178mz/p/8558435.html

cm = confusion_matrix(y_val, y_pred)
#混淆矩陣https://blog.csdn.net/m0_38061927/article/details/77198990
#顯示混淆矩陣
plt.figure(1) #圖名
sns.heatmap(cm, annot=True, fmt='d')#熱力圖
plt.xlabel('Predicted') #軸
plt.ylabel('Truth')

#處理測試test_DATA
test_images = test.values.reshape(-1, 28, 28)#圖片
sc = StandardScaler()
#標準化
test_images = sc.fit_transform(test_images.reshape(-1, 28*28)).reshape(-1, 28, 28)


plt.imshow(test_images[50], cmap='gray')
y_testrfc = rfc.predict(test_images.reshape(-1, 28*28))#預測機
print(y_testrfc[50])

#預測好的TESTDATA寫到sample_submission
submission = pd.DataFrame({'ImageId': np.arange(1, len(y_testrfc)+1), 'Label': y_testrfc})
#轉換pd
submission.to_csv('C:/Users/danie/Desktop/期末報告/sample_submission.csv', index=False)
#寫進去

#print('MSE train:%.3f,test:%.3f'%(mean_squared_error(y_val, y_pred)
#                                 ,mean_squared_error(ans.iloc[:, 1],y_testrfc))) 
#沒給答案不能做MSE
 
fig, ax = plt.subplots(5, 5, figsize=(10, 10)) #印出來
for i in range(5):
    for j in range(5):
        ax[i, j].imshow(test_images[i*5+j], cmap='gray')
        ax[i, j].set_title(y_testrfc[i*5+j])
        ax[i, j].axis('off')




