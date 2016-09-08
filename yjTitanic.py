# -*- coding: utf-8 -*-
#coding: utf-8
import pandas as pd
import numpy as np
from pandas import Series, DataFrame

data_train = pd.read_csv("//home//shekmun//ml//yj//Feature_engineering_and_model_tuning//Kaggle_Titanic//train.csv")
data_train.columns
data_train.info()
data_train.describe()

import matplotlib.pyplot as plt
import matplotlib as mpl
zhfont = mpl.font_manager.FontProperties(fname='/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
fig = plt.figure()
fig.set(alpha = 0.2)

plt.subplot2grid((2,3),(0,0))
data_train.Survived.value_counts().plot(kind = 'bar')
plt.title(u"获救情况 (1为获救)", fontproperties=zhfont)
plt.ylabel(u"人数", fontproperties=zhfont)

plt.subplot2grid((2,3),(0,1))
data_train.Pclass.value_counts().plot(kind = 'bar')
plt.title(u"人数", fontproperties=zhfont)
plt.ylabel(u"乘客等级区分", fontproperties=zhfont)

plt.subplot2grid((2,3),(0,2))
plt.scatter(data_train.Survived, data_train.Age)
plt.ylabel(u"年龄", fontproperties=zhfont)
plt.grid(b=True, which='major', axis='y')
plt.title(u"按年龄看获救分布(1为获救)", fontproperties=zhfont)

plt.subplot2grid((2,3),(1,0), colspan=2)
data_train.Age[data_train.Pclass == 1].plot(kind='kde')
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel(u"年龄", fontproperties=zhfont)
plt.ylabel(u"密度", fontproperties=zhfont)
plt.title(u"各等级的乘客年龄分布", fontproperties=zhfont)
plt.legend((u'first class', u'2 class', u'3 class'),loc='best')

plt.subplot2grid((2,3),(1,2))
data_train.Embarked.value_counts().plot(kind = 'bar')
plt.title(u"各登船口岸上船人数",fontproperties=zhfont)
plt.ylabel(u"人数", fontproperties=zhfont)
plt.show()

#看看各乘客等级的获救情况
fig = plt.figure()
fig.set(alpha=0.2)

Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df=pd.DataFrame({u"survived":Survived_1, u"unsurvived":Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"各乘客等级的获救情况", fontproperties=zhfont)
plt.xlabel(u"乘客等级", fontproperties=zhfont)
plt.ylabel(u"人数", fontproperties=zhfont)
plt.show()

#看看各个登录港口的获救情况
fig = plt.figure()
fig.set(alpha = 0.2)

Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
df = pd.DataFrame({u"survived":Survived_1, u"unsurvived":Survived_0})
df.plot(kind = 'bar', stacked=True)
plt.title(u"各登陆港口的获救情况", fontproperties=zhfont)
plt.xlabel(u"登录港口", fontproperties=zhfont)
plt.ylabel(u"人数", fontproperties=zhfont)
plt.show()

#看看各个性别的获救情况
fig = plt.figure()
fig.set(alpha = 0.2)

Survived_male = data_train.Survived[data_train.Sex == 'male'].value_counts()
Survived_female = data_train.Survived[data_train.Sex == 'female'].value_counts()
df = pd.DataFrame({u"male":Survived_male, u"female": Survived_female})
df.plot(kind = 'bar', stacked = True)
plt.title(u"按性别看获救情况", fontproperties=zhfont)
plt.xlabel(u"性别", fontproperties=zhfont)
plt.ylabel(u"人数", fontproperties=zhfont)
plt.show()

#看看各个舱级别情况下各性别获救情况
fig = plt.figure()
fig.set(alpha=0.65)
plt.title(u"根据舱等级和性别的获救情况",fontproperties=zhfont)

ax1 = fig.add_subplot(141)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar', label="female, high class", color="#FA2479")
ax1.set_xticklabels([u"survived",u"unsurvived"], rotation=0)
ax1.legend([u"famale/high class"], loc="best")

ax2 = fig.add_subplot(142, sharey=ax1)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar', label="female, low class", color="pink")
ax2.set_xticklabels([u"unsurvived",u"survived"], rotation=0)
plt.legend([u"female/low class"],loc='best')

ax3 = fig.add_subplot(143, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar', label = 'male, high class', color="lightblue")
ax3.set_xticklabels([u"unsurvived",u"survived"], rotation=0)
plt.legend([u"male/high class"], loc='best')

ax4 = fig.add_subplot(144, sharey=ax1)
data_train.Survived[data_train.Sex == "male"][data_train.Pclass == 3].value_counts().plot(kind='bar', label='male, low class', color="steelblue")
ax4.set_xticklabels([u"unsurvived",u"survived"], rotation=0)
plt.legend([u"male/low class"],loc='best')
plt.show()

#计算家族的优势，看大家族是否有优势
g = data_train.groupby(['SibSp', 'Survived'])
df = pd.DataFrame(g.count()['PassengerId'])

#计算家里人数多是否有有事
g = data_train.groupby(['Parch','Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
print df
