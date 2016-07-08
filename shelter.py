# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 13:53:23 2016

@author: jining
"""

import pandas as pd
import pylab as pb
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.grid_search import GridSearchCV

#定义一个画图函数；传入的参数为data的某一列
def plotFvR(datac):
    ct = pd.crosstab(datac.loc[0:len(train)-1],train_target)
    ct_tran = ct.div(ct.sum(1),axis=0)
    ct_tran.plot(kind='bar',stacked=True,figsize=(10,10));\
    pb.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0);\
    pb.xticks(rotation=True)


train_path = r'C:\Users\jining\Desktop\机器学习资料\Shelter Animal Outcomes\train.csv\train.csv'
test_path = r'C:\Users\jining\Desktop\机器学习资料\Shelter Animal Outcomes\test.csv\test.csv'
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
train_target = train['OutcomeType']
train_feature = train.drop(['OutcomeType','OutcomeSubtype','AnimalID'],axis=1)
test_feature = test.drop('ID',axis=1)
#将train_feature和test_feature合并
data = pd.concat([train_feature,test_feature],ignore_index=True)
#查看数据info:data.info()
#Name有10916个缺失值
#SexuponOutcome有1个缺失值
#AgeuponOutcome有24个缺失值

#使用Name填补SexuponOutcome
#data[data.Name=='Diego'].SexuponOutcome.value_counts().plot(kind='bar');pb.xticks(rotation=True)
#通过作图发现名字为Diego的宠物Neutered Male占了大多数
#因此用Neutered Male填补缺失值
data.SexuponOutcome.fillna('Neutered Male',inplace=True)

#Name的缺失值使用Unknown表示
data.Name.fillna('Unknown',inplace=True)

#填补AgeuponOutcome
#将Age转换为天
def timeToDays(x):
    if pd.isnull(x):
        return x
    else:
        num,unit = x.split(' ')
        if unit in 'years':
            return int(num)*365
        if unit in 'months':
            return int(num)*30
        if unit in 'weeks':
            return int(num)*7
        if unit in 'days':
            return int(num)
data.AgeuponOutcome = data.AgeuponOutcome.apply(timeToDays)   
#根据SexuponOutcome和AnimalType填补AgeuponOutcome(因为Age的缺失值很少，也可以考虑直接舍弃掉)
imputation=data.groupby(['SexuponOutcome','AnimalType']).median() 
imputation=imputation.AgeuponOutcome.unstack()
for i in range(len(data)):
    if pd.isnull(data.loc[i,'AgeuponOutcome']):
        data.loc[i,'AgeuponOutcome'] = imputation.loc[data.loc[i]['SexuponOutcome']].loc[data.loc[i]['AnimalType']]

#缺失值处理完毕，更新train_feature和test_feature
train_feature = data.loc[0:len(train)-1]        
test_feature = data.loc[len(train):]   
'''     
#作图查看变量AnimalType与target之间的关系;
#cat中的transfer大于dog,dog中的return_to_owner大于cat    
plotFvR(data.AnimalType)        
#作图查看SexuponOutcome与target之间的关系;
#被阉割过的收养率和return率大于其他，而其他的transfer率、死亡率、安乐死率大于阉割过的
plotFvR(data.SexuponOutcome)
'''
#************************************************ 
#将Name更换为knowm与unknown两类
def changeName(x):
    if x!='Unknown':
        return 'Known'
    else:
        return 'Unknown'
data.Name = data.Name.apply(lambda x:changeName(x))

'''      
#Breed和Color的类别取值太多，做特征工程;参考了titanic的ticket处理
BreedCount_Dic = dict(data.Breed.value_counts())
data['BreedGrp'] = data.Breed.apply(lambda x:BreedCount_Dic[x])
a=pd.crosstab(train_target,data.loc[0:len(train)-1,'BreedGrp'])
a=a.T
dic={}
for i in a.index:
    if a.loc[i,:].idxmax()=='Adoption':
        dic[i] = 0
    elif a.loc[i,:].idxmax()=='Return_to_owner':
        dic[i] = 1
    elif a.loc[i,:].idxmax()=='Transfer':
        dic[i] = 2
    elif a.loc[i,:].idxmax()=='Euthanasia':
        dic[i] = 3
    elif a.loc[i,:].idxmax()=='Died':
        dic[i] = 4
data.BreedGrp = data.BreedGrp.apply(lambda x:dic[x])

ColorCount_Dic = dict(data.Color.value_counts())
data['ColorGrp'] = data.Color.apply(lambda x:ColorCount_Dic[x])
a=pd.crosstab(train_target,data.loc[0:len(train)-1,'ColorGrp'])
a=a.T
dic={}
for i in a.index:
    if a.loc[i,:].idxmax()=='Adoption':
        dic[i] = 0
    elif a.loc[i,:].idxmax()=='Return_to_owner':
        dic[i] = 1
    elif a.loc[i,:].idxmax()=='Transfer':
        dic[i] = 2
    elif a.loc[i,:].idxmax()=='Euthanasia':
        dic[i] = 3
    elif a.loc[i,:].idxmax()=='Died':
        dic[i] = 4
data.ColorGrp = data.ColorGrp.apply(lambda x:dic[x])
'''

#Breed中的类别取值太多，根据数据特点将Breed分为mix，pure，dual三类；此种方式证明区分度不是很好
def changeBreed(x):
    if 'Mix' in x or 'mix' in x:
        return 'mix'
    elif '/' in x:
        return 'dual'
    else:
        return 'pure'
data.Breed = data.Breed.apply(lambda x:changeBreed(x))

'''
#去掉Mix;取包含/的第一个值
def changeBreed(x):
    if 'Mix' in x:
        return x.split('Mix')[0].strip()
    elif '/' in x:
        return x.split('/')[0].strip()
    else:
        return x
data.Breed = data.Breed.apply(lambda x:changeBreed(x))
#top15的Breed数量占到了总Breed的近76%；保留76%，剩余的归为other;效果依然不好
dic_Breed = dict(data.Breed.value_counts())
def changeBreed2(x):
    x_num = dic_Breed[x]
    if x_num<323:
        return 'other'
    else:
        return x
data.Breed = data.Breed.apply(lambda x:changeBreed2(x))
'''    

#画图看一下三类特征与最后结果的关系图，检查一下这样划分是否有一定的区分度
#plotFvR(data.Breed)

'''
#Color中的类别取值太多，根据数据特点将Breed分为mix，pure两类
def changeColor(x):
    if '/' in x:
        return 'not'
    else:
        return 'pure'
data.Color = data.Color.apply(lambda x:changeColor(x))
#画图看一下两类特征与最后结果的关系图，检查一下这样划分是否有一定的区分度;结果发现没有区分度
#plotFvR(data.Color)
#1. Color可能与Breed有一定关系，已经考虑了Breed，可以考虑丢弃Color
#2. 重新对Color做特征工程
#3. 暂时把已经做好特征工程的color留在数据里(证明会使模型效果变差)
'''
def changeColor(x):
    if '/' in x:
        return x.split('/')[0].strip()
    else:
        return x
data.Color = data.Color.apply(lambda x:changeColor(x))

#对DateTime做特征工程
#月份、星期几、每天的时间点可能对最终结果有影响
#将DateTime分为年、月、星期几、时
def date2Year(x):
    struct_time = time.strptime(x,'%Y-%m-%d %H:%M:%S')
    return struct_time.tm_year
def date2Month(x):
    struct_time = time.strptime(x,'%Y-%m-%d %H:%M:%S')
    return struct_time.tm_mon
def date2Weekday(x):
    struct_time = time.strptime(x,'%Y-%m-%d %H:%M:%S')
    return struct_time.tm_wday+1
def date2Hour(x):
    struct_time = time.strptime(x,'%Y-%m-%d %H:%M:%S')
    return struct_time.tm_hour
data['Year'] = data.DateTime.apply(lambda x:date2Year(x))
data['Month'] = data.DateTime.apply(lambda x:date2Month(x))
data['Weekday'] = data.DateTime.apply(lambda x:date2Weekday(x))
data['Hour'] = data.DateTime.apply(lambda x:date2Hour(x))
#作图查看Year、Month、Weekday、Hour与结果的对比图
#year对结果没有区分度，应该去掉
#plotFvR(data.Year)
#Month对结果的区分也不大，也去掉
#plotFvR(data.Month)
#Weekday
#周末的adoption比例要大于其他天数
#其他天数的transfer和euthanasia要大于周末
#考虑对Weekday重新编码
#plotFvR(data.Weekday)
#Hour数值比较多，分为工作时间与非工作时间
#工作时间为11~19，其余为非工作时间
#重新转换后区分度比较明显
def changeHour(x):
    if x<19 and x>=11:
        return 'working'
    else: 
        return 'not'
data.Hour = data.Hour.apply(lambda x:changeHour(x))
'''
#另一种hour的转换方式
def changeHour(x):
    if x<12 and x>=6:
        return 'morning'
    elif x<18 and x>=12: 
        return 'afternoon'
    elif x<24 and x>=18:
        return 'night'
    elif x<6 and x>=0:
        return 'midnight'
data.Hour = data.Hour.apply(lambda x:changeHour(x))
'''
#对weekday重新编码
def changeWeekday(x):
    if x in [1,2,3,4,5]:
        return 0
    else:
        return 1
data.Weekday = data.Weekday.apply(lambda x:changeWeekday(x))

#处理SexuponOutcome
def suo2Sex(x):
    if x=='Unknown':
        return 'Unknown'
    else:
        return x.split(' ')[1].strip()
def suo2Status(x):
    if x=='Unknown':
        return 'Unknown'
    else:
        status = x.split(' ')[0].strip()
        if status=='Intact':
            return 'Intact'
        else:
            return 'Neutered'
data['Sex'] = data.SexuponOutcome.apply(lambda x:suo2Sex(x))
data['Status'] = data.SexuponOutcome.apply(lambda x:suo2Status(x))

#基本处理完毕，扔掉不需要的变量
data = data.drop(['DateTime','Month','Color','Year','SexuponOutcome'],axis=1)
#将类别变量转变为int型
le = LabelEncoder()
data.Name = le.fit_transform(data.Name)
data.AnimalType = le.fit_transform(data.AnimalType)
data.Breed = le.fit_transform(data.Breed)
data.Weekday = le.fit_transform(data.Weekday)
data.Hour = le.fit_transform(data.Hour)
data.Sex = le.fit_transform(data.Sex)
data.Status = le.fit_transform(data.Status)

data.Name = data.Name.astype('category')
data.AnimalType = data.AnimalType.astype('category')
data.Breed = data.Breed.astype('category')
data.Weekday = data.Weekday.astype('category')
data.Hour = data.Hour.astype('category')
data.Sex = data.Sex.astype('category')
data.Status = data.Status.astype('category')

#更新trian_feature和test_feature
train_feature = data.loc[0:len(train)-1]        
test_feature = data.loc[len(train):] 

#建立RandomForest模型
clf=RandomForestClassifier(n_estimators=100,min_samples_split=50,max_depth=10,max_features=None)
clf.fit(train_feature,train_target)
cross_val_score(clf,train_feature,train_target,
                cv=KFold(len(train_feature),10,True)).mean()
'''
parameters={'n_estimators':[10,50,100,150,200],
            'max_depth':[5,10,15,20],
            'max_features':['auto','sqrt','log2']
            'min_samples_split':[10,50,100]}
def scoring(clf,x,y):
    scores = []
    for i in range(10):
        score = cross_val_score(clf,x,y,cv=KFold(len(x),10,True)).mean()    
        scores.append(score)
    return sum(scores)/len(scores)
grid_search = GridSearchCV(clf,parameters,scoring=scoring,n_jobs=1,verbose=1)
grid_search.fit(train_feature,train_target)
'''

#预测test_data
prediction = pd.DataFrame(clf.predict_proba(test_feature),
columns=['Adoption','Died','Euthanasia','Return_to_owner','Transfer'],
index=range(1,test_feature.shape[0]+1))

ID = pd.DataFrame(prediction.index,columns=['ID'],index=prediction.index)
prediction = pd.concat([ID,prediction],axis=1)

prediction.to_csv(r'C:\Users\jining\Desktop\机器学习资料\Shelter Animal Outcomes\prediction.csv', index=False) 






