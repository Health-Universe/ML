###以CatBoostClassifier模型开发web应用#################
# 导入所需的包
import pandas as pd
import catboost
from catboost import CatBoostClassifier
#Catboost采用排序提升的方法对抗训练集中的噪声点，从而避免梯度估计的偏差，进而解决预测偏移的问题
#pip install catboost
# 导入数据
X= pd.read_csv("D:/Documents/Thrombolysis/second/X_train.csv",index_col=0)
X.info()
# 导入数据
y= pd.read_csv("D:/Documents/Thrombolysis/second/y_train.csv",index_col=0)

y.info()
# 特征变量提取及转换
#X = df[["Height", "Weight", "Eye"]]
#X = X.replace(["Brown", "Blue"], [1, 0])

# 分类变量提取
#y = df["Species"]

# 建模
clf = CatBoostClassifier()
clf.fit(X, y)
####建模过程同上，不用再重复
import joblib

joblib.dump(clf, "D:/Documents/Thrombolysis/second/clf1.pkl")