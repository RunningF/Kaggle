# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 09:17:31 2016

Kaggle Titanic Data Exploration

@author: jining
"""

import numpy as np
import pandas as pd
import pylab as pb
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import pearsonr,pointbiserialr, spearmanr
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
#载入训练数据
data = pd.read_csv(r'C:\Users\jining\Desktop\机器学习资料\train.csv')
#查看数据的头和尾
data.head()
data.tail()
#查看数据各个feature的数据类型,object表示字符串类型
data.dtypes
#查看数据的一些基本信息(有哪些feature、多少样本量、各feature缺失值情况、各feature的数据类型)
data.info()
#查看数值型feature的统计特征
data.describe()

#作图了解数据*************************************
#查看类别变量的数据情况
pb.figure(1,figsize=(10,10));\
pb.subplot(321);data.Survived.value_counts().plot(kind='bar',title='Survived',alpha=0.5);pb.grid();pb.xticks(rotation=True);\
pb.subplot(322);data.Sex.value_counts().plot(kind='bar',title='Sex',alpha=0.5);pb.grid();pb.xticks(rotation=True);\
pb.subplot(323);data.Embarked.value_counts().plot(kind='bar',title='Embarked',alpha=0.5);pb.grid();pb.xticks(rotation=True);\
pb.subplot(324);data.Pclass.value_counts().plot(kind='bar',title='Pclass',alpha=0.5);pb.grid();pb.xticks(rotation=True);\
pb.subplots_adjust(wspace=0.25, hspace=0.5);pb.xticks(rotation=True)
#查看数值类型的数据情况
pb.figure(2,(10,10));\
pb.subplot(221);data.Age.hist();pb.title('Age',fontsize=20);pb.xlabel('Value',size=14);pb.ylabel('Frequency',size=14);\
pb.subplot(222);data.SibSp.hist();pb.title('SibSp',fontsize=20);pb.xlabel('Value',size=14);pb.ylabel('Frequency',size=14);\
pb.subplot(223);data.Parch.hist();pb.title('Parch',fontsize=20);pb.xlabel('Value',size=14);pb.ylabel('Frequency',size=14);\
pb.subplot(224);data.Fare.hist();pb.title('Fare',fontsize=20);pb.xlabel('Value',size=14);pb.ylabel('Frequency',size=14)
#查看各类别变量对Survived的影响
#Pclass
pclass_ct = pd.crosstab(data.Pclass,data.Survived)
pclass_ct_tran = pclass_ct.div(pclass_ct.sum(1),axis=0)
pclass_ct_tran.plot(kind='bar',stacked=True);pb.title('Survival Rate by Passenger Classes',fontsize=20);pb.xlabel('Passenger Class',size=14);pb.ylabel('Survival Rate',size=14);pb.xticks(rotation=True);pb.grid()
#Sex
sex_ct = pd.crosstab(data.Sex,data.Survived)
sex_ct_tran = sex_ct.div(sex_ct.sum(1),axis=0)
sex_ct_tran.plot(kind='bar',stacked=True);pb.title('Survival Rate by Passenger Sex',fontsize=20);pb.xlabel('Passenger Sex',size=14);pb.ylabel('Survival Rate',size=14);pb.xticks(rotation=True);pb.grid()
#Pclass and Sex
#female,Pclass,Survived
female_df = data[data.Sex=='female']
female_df_ct = pd.crosstab(female_df.Pclass,female_df.Survived)
female_df_ct_tran = female_df_ct.div(female_df_ct.sum(1),axis=0)
female_df_ct_tran.plot(kind='bar',stacked=True);pb.title('Female Survival Rate by Pclass',fontsize=20);pb.xlabel('Passenger Pclass',size=14);pb.ylabel('Survival Rate',size=14);pb.xticks(rotation=True);pb.grid()
#male,Pclass,Survived
male_df = data[data.Sex=='male']
male_df_ct = pd.crosstab(male_df.Pclass,male_df.Survived)
male_df_ct_tran = male_df_ct.div(male_df_ct.sum(1),axis=0)
male_df_ct_tran.plot(kind='bar',stacked=True);pb.title('Male Survival Rate by Pclass',fontsize=20);pb.xlabel('Passenger Pclass',size=14);pb.ylabel('Survival Rate',size=14);pb.xticks(rotation=True);pb.grid()
#Embarked
#Embarked中存在missing value,先进行imputation
data.Embarked[pd.isnull(data.Embarked)]='S'
Embarked_ct = pd.crosstab(data.Embarked,data.Survived)
Embarked_ct_tran = Embarked_ct.div(Embarked_ct.sum(1),axis=0)
Embarked_ct_tran.plot(kind='bar',stacked=True);pb.title('Survival Rate by Passenger Embarked',fontsize=20);pb.xlabel('Passenger Embarked',size=14);pb.ylabel('Survival Rate',size=14);pb.xticks(rotation=True);pb.grid()
'''
#将Embarked转换为数值的方法
Emb_sort_unique = sorted(data.Embarked.unique(),key=lambda x:str(x))
Emb_dict = dict(zip(Emb_sort_unique,range(len(Emb_sort_unique))))
data['Embarked_val'] = data.Embarked.map(Emb_dict)
'''
#Age与suvived直方图
data.Age[pd.isnull(data.Age)]=data.Age.median()
df1=data.Age[data.Survived==0]
df2=data.Age[data.Survived==1]
pb.hist([df1,df2],bins=max(data.Age)/10,stacked=True,range=(1,max(data.Age)));pb.grid();pb.legend(('Died','Survived'),loc='best')
#Age与Pclass概率密度图
df1=data.Age[data.Pclass==1]
df2=data.Age[data.Pclass==2]
df3=data.Age[data.Pclass==3]
df1.plot(kind='kde');df2.plot(kind='kde');df3.plot(kind='kde');pb.legend(('1','2','3'),loc='best')
#FamilySize
data['FamilySize']=data.SibSp+data.Parch
familysize_ct = pd.crosstab(data.FamilySize,data.Survived)
familysize_ct_tran = familysize_ct.div(familysize_ct.sum(1),axis=0)
familysize_ct_tran.plot(kind='bar',stacked=True);pb.grid()

#*****************************************
#将类别数据特征因子化
data['Embarked_C']=pd.get_dummies(data.Embarked)['C']
data['Embarked_Q']=pd.get_dummies(data.Embarked)['Q']
data['Embarked_S']=pd.get_dummies(data.Embarked)['S']
data['Is_female']=pd.get_dummies(data.Sex)['female']
#去掉不用的特征
data=data.drop(['Name','Ticket','PassengerId','SibSp','Parch','Embarked','Sex','Cabin'],axis=1)

#*****************************************
#提取ticket前缀的函数
def tickit_prefix(s):
    s=s.split(' ')[0].strip()
    if s.isdigit():
        return 'NoClue'
    else:
        return s
        
def clean_data(df):
    #用df的PassengerId作为index
    df.index = df.PassengerId  
    
    #填补缺失值
    df.Fare[pd.isnull(df.Fare)]=df.Fare.median()    
    df.Age[pd.isnull(df.Age)]=df.Age.median()
    df.Embarked[pd.isnull(df.Embarked)]='S'
    
    #进行一点特征工程：
    #将两个特征合并为一个特征
    df['FamilySize'] = df.SibSp + df.Parch
    #将Name分解为title
    df['title']=df.Name.apply(lambda x:x.split(',')[1].split('.')[0].strip())
    Title_Dictionary={'Capt': 'Officer','Col': 'Officer','Don': 'Royalty','Dona': 'Royalty',
     'Dr': 'Officer','Jonkheer': 'Royalty','Lady': 'Royalty','Major': 'Officer',
     'Master': 'Master','Miss': 'Miss','Mlle': 'Miss','Mme': 'Mrs','Mr': 'Mr',
     'Mrs': 'Mrs','Ms': 'Mrs','Rev': 'Officer','Sir': 'Royalty','the Countess': 'Royalty'}
    df['title']=df.title.apply(lambda x:Title_Dictionary[x])
    #将Ticket的前缀取出来
    df['TicketPrefix']=df.Ticket.apply(lambda x:tickit_prefix(x))
    TPUnique={'A./5.','A.5.','A/4','A/4.','A/5','A/5.','A/S','A4.','C','C.A.',
              'C.A./SOTON','CA','CA.','F.C.','F.C.C.','Fa','LINE','NoClue',
              'P/PP','PC','PP','S.C./A.4.','S.C./PARIS','S.O./P.P.','S.O.C.',
              'S.O.P.','S.P.','S.W./PP','SC','SC/AH','SC/PARIS','SC/Paris',
              'SCO/W','SO/C','SOTON/O.Q.','SOTON/O2','SOTON/OQ','STON/O',
              'STON/O2.','SW/PP','W./C.','W.E.P.','W/C','WE/P'}
    #类别数据特征因子化
    df['Embarked_C']=pd.get_dummies(df.Embarked)['C']
    df['Embarked_Q']=pd.get_dummies(df.Embarked)['Q']
    df['Embarked_S']=pd.get_dummies(df.Embarked)['S']
    df['Pclass_1']=pd.get_dummies(df.Pclass)[1]
    df['Pclass_2']=pd.get_dummies(df.Pclass)[2]
    df['Pclass_3']=pd.get_dummies(df.Pclass)[3]
    df['Is_female']=pd.get_dummies(df.Sex)['female']
    
    for i in pd.get_dummies(df.title).columns:
        df['title_'+i]=pd.get_dummies(df.title)[i] 
        
    for i in set(TPUnique).intersection(pd.get_dummies(df.TicketPrefix).columns):
        df['TicketPrefix_'+i]=pd.get_dummies(df.TicketPrefix)[i] 
        
    for i in set(TPUnique).difference(pd.get_dummies(df.TicketPrefix).columns):
        df['TicketPrefix_'+i]=pd.Series(np.zeros(len(df)),index=df.index)
        
    
    #扔掉不要的特征
    df=df.drop(['Name','Ticket','PassengerId','Embarked','Sex','Cabin','Pclass','title','TicketPrefix'],axis=1)
    
    return df
   
#*****************************************
#random forest training
clf = RandomForestClassifier(n_estimators=100)
data_feature = data.iloc[:,1:]
data_target = data.iloc[:,0]
cross_val_score(clf,data_feature,data_target,cv=10).mean()

#random forest predicting
clf.fit(data_feature,data_target)
test_data = pd.read_csv(r'C:\Users\jining\Desktop\机器学习资料\test.csv')
test_data = clean_data(test_data)
prediction = clf.predict(test_data)
saved_data = pd.DataFrame({'PassengerId':test_data.index,'Survived':prediction})
saved_data.to_csv(r'C:\Users\jining\Desktop\机器学习资料\prediction3+.csv',index=False)

#feature selection
def get_sort_abs_cor(data):
    columns=data.columns.values
    correlation=[]
    #spearmanr计算类别变量之间的相关性
    #pointbiserialr计算类别变量与连续变量之间的相关性
    for i in columns:
        if len(data[i].unique())<=2:
            correlation.append(spearmanr(data['Survived'],data[i])[0])
        else:
            correlation.append(pointbiserialr(data['Survived'],data[i])[0])
        
    cor=pd.DataFrame({'Correlation':correlation})
    cor.index=columns
    cor['abs_cor']=cor.Correlation.apply(lambda x:abs(x))
    cor=cor.iloc[1:,:]
    sort_abs_cor=cor.abs_cor.sort_values(ascending=False)
    
    return sort_abs_cor
#将sort_abs_cor中的特征按顺序累计生成决策树,获得决策树得分，画图
score=[]
for i in range(len(sort_abs_cor)):
    x=data[sort_abs_cor.index[0:i+1]]
    y=data.Survived
    clf = DecisionTreeClassifier()
    score.append(cross_val_score(clf,x,y,cv=10).mean())
pb.figure(figsize=(10,10));pb.plot(range(len(sort_abs_cor)),score,'bo-');pb.grid()
#选择得分最高的点所对应的特征数量，做特征选择
x=data[sort_abs_cor.index[0:12].values]
y=data.Survived
clf = RandomForestClassifier(n_estimators=100)
clf.fit(x,y)
clf.score(x,y)

#模型参数调优
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=True)  
clf=RandomForestClassifier()
parameters = {'n_estimators':[1,5,10,15,20],'criterion':['gini','entropy'],'max_features':['auto','sqrt','log2',None],'max_depth':[1,5,10,15,20],'min_samples_split':[1,5,10,15,20],'min_weight_fraction_leaf':[0.1,0.2,0.3,0.4,0.5],'max_leaf_nodes':[2,5,10,15,20]}
grid_search = GridSearchCV(clf,parameters,n_jobs = 1,verbose=1)
grid_search.fit(x_train,y_train)
grid_search.best_params_
