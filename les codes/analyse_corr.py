import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:\\untitled2\\venv\\HR.csv')
#利用交叉分析方法来分析各个部门的离职率关系 使用独立p检验 得到各部门的独立分布
sl_s=df['salary']

df=df.where(sl_s!='nme').dropna()

#得到的是一个字典 字典的键是department的名称 值是dataframe中index，即所在部门的具体行的数字
df_indices=df.groupby(by='department').indices
#取出sales部门的离职率的值 是array对象 一组数组 通过iloc得到行的index

sales_values=df['left'].iloc[df_indices['sales']].values
technical_values=df['left'].iloc[df_indices['technical']].values
#采用t检验 看两个分布的联系
print(ss.ttest_ind(sales_values,technical_values))
#两两求p值：取出keys 再初始化一个矩阵 np.zeros 创建为0矩阵
#得到字典keys的集合 用list将其转换为数组 不然得不到其长度无法生成后续的矩阵
df_keys=list(df_indices.keys())
print(type(df_keys))
df_t_mat=np.zeros([len(df_keys),len(df_keys)])
#形成二维数组，将department里面的各部门两两比较
for i in range(len(df_keys)):
    for j in range(len(df_keys)):
        p_value=ss.ttest_ind(df['left'].iloc[df_indices[df_keys[i]]].values,
        df['left'].iloc[df_indices[df_keys[j]]].values)[1]
        #加上判断条件 加上p值小于0.05 将矩阵=-1 则颜色越深差异越大
        df_t_mat[i][j]=p_value

sns.heatmap(df_t_mat,xticklabels=df_keys,yticklabels=df_keys)
plt.show()
#使用data生成透视表来做交叉分析
#指定dataframe，关注的是 left，横坐标是promotion和salarys，即每行的开头。列是work_accident，即每列的开头
#得到的结果即 工作晋升 工资 工作失误的混合矩阵 值是离职率，可以看出特殊值
#聚合方法 离职率 所以选mean（）填入函数
piv_tb=pd.pivot_table(df,values='left',index=['promoton_last_5years','salary'],columns=['Work_accident'],
                    aggfunc=np.mean )

print(piv_tb)
#用seanborn画图 颜色越深离职率越高
sns.heatmap(piv_tb,vmin=0,vmax=1)

#分组分析
#离散值的分组sns.barplot(x='salary',y='left',hue='department',data=df)

#连续值的分组 得到的长度有拐弯的点 则以这一点做界限
#st_s=df['satisfaction_level']
#sns.barplot(list(range(len(st_s))),st_s.sort_values())

#用heatmap来显示色块 根据色块深浅可以得到正相关或者负相关以及不相关
sns.set(font_scale=1.5)
sns.heatmap(df.corr(),vmin=-1,vmax=1,cmap=sns.color_palette('RdBu',n_colors=128))
plt.show()

#熵的代码实现 通过values将grouby转换成array对象
s1=pd.Series(['x1','x1','x2','x2','x2','x2'])
s2=pd.Series(['y1','y1','y1','y2','y2','y2'])
def getentropy(s):
    prt_ary=pd.groupby(s,by=s).count().values/float(len(s))
    return (np.log2(prt_ary)*prt_ary).sum()
#条件熵
def getCondEntropy(s1,s2):
    d=dict()
    for i in list(range(len(s1))):
        #一个结构体 key是s1的值，值是s1值下 s2的分布
        d[s1[i]]=d.get(s1[i],[])+ [s2[i]]
    for k in d:
        result=+getentropy(d[k])*len(d([k]))/float(len(s1))
    return result
#互信息：作为条件变量的信息熵-这个条件下的y分布的条件熵
def getentropyGain(s1,s2):
    return getentropy(s2)-getCondEntropy(s1,s2)
#熵增益率： 互信息/作为条件变量的信息熵
def getEntropyGainRatio(s1,s2):
    return getentropyGain(s1,s2)/getentropy(s2)
#分析离散属性的相关性：
import math
def getDiscteteCorr(s1,s2):
    return getentropyGain(s1,s2)/math.sqrt(getentropy(s1)*getentropyGain(s2))
#gini系数
#先定义求熵的平方和的函数
def getProsss(s):
    if not isinstance(s,pd.core.series.Series):
        s=pd.Series(s)
    prt_ary = pd.groupby(s,by=s).count.values / float(len(s))
    return sum(prt_ary**2)
def getGini(s1,s2):
    d = dict()
    for i in list(range(len(s1))):
        d[s1[i]] = d.get(s1[i], []) + [s2[i]]
    return 1-sum([getProsss(d[k]*len(d[k])/float(len(s1)) for k in d )])

