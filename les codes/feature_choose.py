import numpy as np
import pandas as pd
import scipy.stats as ss
df=pd.DataFrame({'A':ss.norm.rvs(size=10),'B':ss.norm.rvs(size=10),
                'C':ss.norm.rvs(size=10),'D':np.random.randint(low=0,high=2,size=10 )})
#引入svr回归器
from sklearn.svm import SVR
#引入决策树
from sklearn.tree import DecisionTreeRegressor
#特征
x=df.loc[:,['A','B','C']]
print(x)
#标注
y=df.loc[:,'D']


#引入特征选择的包 selectKbest是过滤思想，RFE是包裹思想，SelectFromModel是嵌入思想
from sklearn.feature_selection import SelectKBest,RFE,SelectFromModel

#过滤思想 SelectKbest可以自己选择函数 通常一张表会通过几种函数来对比
skb=SelectKBest(k=2)

#再调用函数之前需要先用fit拟合 即找到训练集x的均值，方差，最大值等固有属性
skb.fit(x,y)

#进行数据转换 包括标准化等操作，看具体使用的函数要求 。得到的结果与原来的df对比则可以得到保留的列

print(skb.transform(x))

#对于某些estimator的选择：有约束，必须有feature_importance 或者coef这两个参数

#RFE包裹思想 通过遍历子集来去除特征
# 需要制定estimator,通过svr回归器来建立,linear复杂度不高,第二个参数表示最终选择的特征，step表示每次迭代删除的特征
rfe = RFE(estimator=SVR(kernel='linear',n_features_to_selcet=2,step=1))
#拟合过后进行变换，即包含了fit又包含了转换
print(rfe.fit_transform(x))

#嵌入思想 评价指标通过决策树生成 threshold表示重要性因子，即阈值。即低于0.1则会被去掉
sfm= SelectFromModel(estimator=DecisionTreeRegressor(),threshold=0.1)
print(sfm.fit_transform(x,y))