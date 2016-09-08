# -*- coding: utf-8 -*-
#coding: utf-8
import pandas as pd
import numpy as np
from pandas import Series, DataFrame

data_train = pd.read_csv("E:\\ml\\3\\Feature_engineering_and_model_tuning\\Feature_engineering_and_model_tuning\\Kaggle_Titanic\\train.csv")
data_train.columns
data_train.info()
data_train.describe()

import matplotlib.pyplot as plt

fig = plt.figure()
fig.set(alpha = 0.2)

plt.subplot2grid((2,3),(0,0))
data_train.Survived.value_counts().plot(kind = 'bar')
plt.title(u"获救情况 (1为获救)")
plt.ylabel(u"人数")

plt.subplot2grid((2,3),(0,1))
data_train.Pclass.value_counts().plot(kind = 'bar')
plt.title(u"人数")
plt.ylabel(u"乘客等级区分")

plt.subplot2grid((2,3),(0,2))
plt.scatter(data_train.Survived, data_train.Age)
plt.ylabel(u"年龄")
plt.grid(b=True, which='major', axis='y')
plt.title(u"按年龄看获救分布(1为获救)")

plt.subplot2grid((2,3),(1,0), colspan=2)
data_train.Age[data_train.Pclass == 1].plot(kind='kde')
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel(u"年龄")
plt.ylabel(u"密度")
plt.title(u"各等级的乘客年龄分布")
plt.legend((u'头等舱', u'2等舱', u'3等舱'), loc='best')

plt.subplot2grid((2,3),(1,2))
data_train.Embarked.value_counts().plot(kind = 'bar')
plt.title(u"各登船口岸上船人数")
plt.ylabel(u"人数")
plt.show()

#看看各乘客等级的获救情况
fig = plt.figure()
fig.set(alpha=0.2)

Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df=pd.DataFrame({u"获救":Survived_1, u"未获救":Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"各乘客等级的获救情况")
plt.xlabel(u"乘客等级")
plt.ylabel(u"人数")
plt.show()

#看看各个登录港口的获救情况
fig = plt.figure()
fig.set(alpha = 0.2)

Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
df = pd.DataFrame({u"获救":Survived_1, u"未获救":Survived_0})
df.plot(kind = 'bar', stacked=True)
plt.title(u"各登陆港口的获救情况")
plt.xlabel(u"登录港口")
plt.ylabel(u"人数")
plt.show()

#看看各个性别的获救情况
fig
