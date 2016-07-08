# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 08:46:24 2016

@author: jining
"""

import numpy as np
import pandas as pd
import pylab as pb
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV

#读入training data和test data
data_train = pd.read_csv(r'C:\Users\jining\Desktop\机器学习资料\train.csv')
data_test = pd.read_csv(r'C:\Users\jining\Desktop\机器学习资料\test.csv')

#将data_train分为feature_train和target_train
feature_train = data_train.drop(['Survived'],axis=1)
target_train = data_train.Survived
feature_test = data_test

#将训练数据与测试数据合并
data_comb=pd.concat([feature_train,feature_test],ignore_index=True)

#缺失值处理
#通过data_comb.count()发现训练集中包含缺失值的变量有Age,Cabin,Embarked,Fare

#填补Embarked
#1.作图发现Embarked中S占了大多数，可以直接用S对缺失值进行填补
#2.考虑到Embarked也许和Pclass、Fare有关系，尝试能否用Pclass和Fare对Embarked进行预测
#这里采用第二种方法
#查看Embarded缺失值得情况,发现pclass均为1，Fare均为80
Embarked_Pclass_Fare=data_comb[['Embarked','Pclass','Fare']]
Embarked_Pclass_Fare[Embarked_Pclass_Fare.Embarked.isnull()]
#做Fare关于Embarked和Pclass的箱线图(一会再看)
df1=data_comb.Fare[data_comb.Embarked=='S'][data_comb.Pclass==1]
df2=data_comb.Fare[data_comb.Embarked=='S'][data_comb.Pclass==2]
df3=data_comb.Fare[data_comb.Embarked=='S'][data_comb.Pclass==3]
df4=data_comb.Fare[data_comb.Embarked=='C'][data_comb.Pclass==1]
df5=data_comb.Fare[data_comb.Embarked=='C'][data_comb.Pclass==2]
df6=data_comb.Fare[data_comb.Embarked=='C'][data_comb.Pclass==3]
df7=data_comb.Fare[data_comb.Embarked=='Q'][data_comb.Pclass==1]
df8=data_comb.Fare[data_comb.Embarked=='Q'][data_comb.Pclass==2]
df9=data_comb.Fare[data_comb.Embarked=='Q'][data_comb.Pclass==3]
pb.boxplot([df1,df2,df3,df4,df5,df6,df7,df8,df9]);pb.grid();pb.ylim(0,300);pb.plot((0,10),(80,80),'g--')
#通过箱线图可以发现Fare==80刚好是pclass==1，Embarked==C的中位线附近，所以用C来填补Embarked
data_comb.Embarked=data_comb.Embarked.fillna('C')

#填补Cabin
#Cabin的缺失值比较多，这里把缺失值转换为'UNK',means unknown
data_comb.Cabin=data_comb.Cabin.fillna('UNK')

#填补Age，稍后再说


#特征工程
#对Name做特征工程
#存活率的高低也许和此人的社会地位有关，将Name一列分解为Title和Surname
#data_comb['Surname']=data_comb.Name.apply(lambda x:x.split(',')[0])
data_comb['Title']=data_comb.Name.apply(lambda x:x.split(',')[1].strip().split('.')[0].strip())
#根据Title_Dictionary对Title中的值进行映射
Title_Dictionary = {
"Capt": "Officer",
"Col": "Officer",
"Major": "Officer",
"Jonkheer": "Sir",
"Don": "Sir",
"Sir" : "Sir",
"Dr": "Dr",
"Rev": "Rev",
"the Countess": "Lady",
"Dona": "Lady",
"Mme": "Mrs",
"Mlle": "Miss",
"Ms": "Mrs",
"Mr" : "Mr",
"Mrs" : "Mrs",
"Miss" : "Miss",
"Master" : "Master",
"Lady" : "Lady"} 
data_comb.Title = data_comb.Title.apply(lambda x:Title_Dictionary[x])
def title_label(x):
    if x in ['Sir','Lady']:
        return 'Royalty'
    elif x in ['Dr','Officer','Rev']:
        return 'Officer'
    else:
        return x
#继续映射
data_comb.Title = data_comb.Title.apply(title_label)
#对Title映射的处理可以参考下面FamilySize的方式，以后尝试
#pd.crosstab(data_comb.Title[0:891],target_train).plot(kind='bar',stacked=True);pb.grid()


#根据SibSp和Parch新增一个特征变量FamilySize,+1是把自己也算进去
data_comb['FamilySize'] = data_comb.SibSp + data_comb.Parch +1
#作图考察FamilySize与Survived之间的关系
pd.crosstab(data_comb.FamilySize[0:891],target_train).plot(kind='bar',stacked=True);pb.grid()
#从图中可以看出FamilySize==2,3,4的家庭存活率要大于死亡率，
#FamilySize==1,5,6,7的家庭存活率要低于死亡率
#而FamilySize>7的家庭存活率为0
#因此，根据以上观察重新映射FamilySize
def familysize_label(x):
    if x in [2,3,4]:
        return 2
    elif x in [1,5,6,7]:
        return 1
    elif x>7:
        return 0
data_comb.FamilySize = data_comb.FamilySize.apply(familysize_label)       


#根据Ticket做特征工程
#将ticket中的多余字符全部去掉，然后统计不同ticket的数量,以此作为TicketGrp
def tix_clean(j):
    j = j.replace(".", "")
    j = j.replace("/", "")
    j = j.replace(" ", "")
    return j
data_comb.Ticket = data_comb.Ticket.apply(tix_clean)
TicketCount_Dic = dict(data_comb.Ticket.value_counts())
data_comb['TicketGrp'] = data_comb.Ticket.apply(lambda x:TicketCount_Dic[x])
#作图查看TicketGrp与Survived之间的关系
pd.crosstab(data_comb.TicketGrp[0:891],target_train).plot(kind='bar',stacked=True);pb.grid()
#从图中可以看出TicketGrp==2,3,4时存活率要大于死亡率
#TicketGrp==1,5,6,7,8时存活率小于死亡率
#TicketGrp>8时存活率为0
#因此，根据以上观察重新映射TicketGrp
def Ticket_label(x):
    if x in [2,3,4]:
        return 2
    elif x in [1,5,6,7,8]:
        return 1
    elif x>8:
        return 0
data_comb.TicketGrp = data_comb.TicketGrp.apply(Ticket_label)


#根据Cabin做特征工程
#Cabin的数据形式是一个字母加一串数字，字母有可能是甲板的等级，数字有可能是房间号码
#这里考虑字母，也就是甲板，将字母提取出来
data_comb['Deck'] = data_comb.Cabin.apply(lambda x:x[0])



#特征工程完毕，丢掉不需要的feature
data_comb=data_comb.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin'],axis=1)


#对Age进行缺失值补全
#根据Title和Pclass对Age进行补全
#分析没有缺失的Age和Title、Pclass之间的关系
Age_Title_Pclass = data_comb.loc[data_comb.Age.notnull(),['Age','Title','Pclass']]
age_fill = Age_Title_Pclass.groupby(by=['Title','Pclass']).median()
age_fill = age_fill.Age.unstack()
#根据未缺失的Age对缺失的Age进行填补
for i in range(len(data_comb)):
    if pd.isnull(data_comb.loc[i,'Age']):
        data_comb.loc[i,'Age']=age_fill.loc[data_comb.loc[i,'Title'],data_comb.loc[i,'Pclass']]

#对Fare进行补全
#查看带有缺失值的Fare的情况
data_comb[data_comb.Fare.isnull()]
#根据Pclass和Embarked对Fare进行补缺
fare_fill = data_comb.groupby(by=['Pclass','Embarked']).median()
fare_fill = fare_fill.Fare.unstack() 
data_comb.Fare=data_comb.Fare.fillna(fare_fill.loc[3,'S'])


#数据补全完成
#把类别变量变为字符串类型
'''
data_comb.Pclass = data_comb.Pclass.astype('object')
data_comb.FamilySize = data_comb.FamilySize.astype('object')
data_comb.TicketGrp = data_comb.TicketGrp.astype('object')
'''
#将类别变量变为dummy variable
data_comb = pd.get_dummies(data_comb)
#将连续变量标准化(先不用)

df_train = data_comb.iloc[0:len(feature_train),:]    
df_test = data_comb.iloc[len(feature_train):,:]
df_target = target_train
    
    
#特征选择与模型参数选择
select = SelectKBest()
clf = RandomForestClassifier(random_state = 10,warm_start = True)
pipeline=Pipeline([('select',select),('clf',clf)])  
#用gridsearch进行调参，GridSearchCV中的scoring参数应当换为自己自定义的
parameters={'select__k':[20,21],'clf__n_estimators':[24,26],
            'clf__max_depth':[5,6],'clf__max_features':['auto','sqrt','log2',None]}
grid_search = GridSearchCV(pipeline,parameters,n_jobs = 1,verbose=1)
grid_search.fit(df_train,df_target)
#查看最优参数
grid_search.best_params_

#使用最优参数进行模型训练
select = SelectKBest(k=20)
clf = RandomForestClassifier(random_state=10,warm_start=True,
                             n_estimators=26,max_depth=6,max_features='sqrt')
pipeline=Pipeline([('select',select),('clf',clf)])  
pipeline.fit(df_train, df_target)     
cv_score = cross_val_score(pipeline, df_train, df_target, cv= 10)
prediction = pipeline.predict(df_test)
submission = pd.DataFrame({"PassengerId":data_test["PassengerId"],"Survived":prediction})
submission.to_csv(r'C:\Users\jining\Desktop\机器学习资料\prediction.csv', index=False) 




'''
pipeline=Pipeline([('select',select),('clf',clf)])
scores=[]
for i in range(1,26):
    pipeline.set_params(select__k=i)
    scores.append(cross_validation.cross_val_score(pipeline,df_train,df_target,cv=10).mean())
pb.plot(range(1,26),scores,'r.-');pb.grid();pb.xlim(1,25)
'''
'''
select = SelectKBest(k = 20)
select.fit(df_train,df_target)
score_dict = dict(zip(select.scores_,df_train.columns))
score_dict_sort = sorted(d.items(),key=lambda x:x[0],reverse=True)
'''

'''
def title_lebel(x):
    if x =='Lady':
        return 3
    elif x in ['Master','Miss','Mrs']:
        return 2
    elif x in ['Dr','Mr','Officer','Sir']:
        return 1
    elif x =='Rev':
        return 0
data_comb.Title = data_comb.Title.apply(title_lebel)


def tickit_label(x):
    if x.isdigit():
        return 'N'
    else:
        return x[0]
data_comb.Ticket = data_comb.Ticket.apply(tickit_label)
def tickit_label_2(x):
    if x in ['P','F']:
        return 1
    else:
        return 0
data_comb.Ticket = data_comb.Ticket.apply(tickit_label_2)

def deck_label(x):
    if x in ['B','C','D','E','F']:
        return 2
    elif x=='T' or x=='G':
        return 0
    else:
        return 1
data_comb.Deck=data_comb.Deck.apply(deck_label)
      
ACFLPSW