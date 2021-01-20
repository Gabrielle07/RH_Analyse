import numpy as np
import matplotlib.pyplot as plt
#定义样本点
n_samples=1000
#生成样本点
from sklearn.datasets import make_circles,make_blobs,make_moons
#聚类算法
from sklearn.cluster import KMeans,DBSCAN,AgglomerativeClustering


# 生成4个不同形状的数据集：进行Kmeans实验

#指定circle里的点，noise代表噪声比例
#circles里面第一部分是生成的点，第二部分标注
#factor 就是小圆与大圆的间隔
circles=make_circles(n_samples=n_samples,factor=0.5,noise=0.05)
moons=make_moons(n_samples=n_samples,noise=0.05)
#random_state防止产生的pyplot有位置变化
#当聚类的超过颜色的下标时，画图会报错 调节center_box参数重构样本 放下数据尺度 防止聚类值过于分散
#调节cluster 标准差可以防止过于集聚
blobs=make_blobs(n_samples=n_samples,random_state=8,center_box=(-1,1),cluster_std=0.1)
#生成随机值，2维数据，增加None参数，则circles里面的第二部分的标注不会显示
random_data=np.random.rand(n_samples,2),None
#增加散点图下颜色的集合 不同类别的数据将用不同颜色表示
colors='bgrcmk'
#确定随机数据集
data=[circles,moons,blobs,random_data]


#确定模型 第一个作为None即画出原始数据图形 并不进行训练
#可以在models里面增加训练器，每一个实体models占一行的位置，并按照此模型的特点对数据进行聚类画图
models=[('None',None),('Kmeans',KMeans(n_clusters=3)),
        #指定邻域和最小密度 基于密度
        ('DBSCAN',DBSCAN(min_samples=3,eps=0.2)),
        #指定聚类的距离衡量方式 基于方差
        ('Agglomeative',AgglomerativeClustering(n_clusters=3,linkage='ward'))]

#轮廓系数评价指标
from sklearn.metrics import silhouette_score

#画图
f=plt.figure()
# enumerate函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列
#遍历，将每一个数据集用不同的模型训练，inx为模型下标，用来确定画图位置，clt为模型实体
for inx,clt in enumerate(models):
    #模型名字与模型实体，模型分为两部分，可以直接赋值
    #模型名字用作画图的title，模型实体部分用来得到标注以及训练数据集
    clt_name,clt_entity=clt

    for i ,dataset in enumerate(data):
        # 用enumerate将 data分成两部分 i是下标，dataset是数据
        #将dataset分成两部分，第一部分是数组，第二部分是原始的标注（即用来画图） 没有用
        #X用来作为数据训练集
        X,Y=dataset
        #如果实体为空
        if not clt_entity:
            #X 即第一部分的数组, 模型最终的输出结果设为 clt_res,如果不进行训练，即model是None的时候，则直接赋值标注都为0
            clt_res=[0 for item in range(len(X))]

        #如果有加入模型，如kmeans训练器，则可以用kmeans去拟合X
        #将clt_entity模型实体来训练之前的dataset里面的数据集
        #并且得到聚类后的label赋值给clt_res，作为np.int输出
        else:
            clt_entity.fit(X)
            clt_res=clt_entity.labels_.astype(np.int)

        #画图部分 每一个数据集在对应的模型下产生一个图

        #inx是模型的下标，即一个模型对应的图形占一行
        #列则是原始dataset里面不同的数据集
        #增加子图，确定整个大图的行数以及列数，行数是models的数量，列数是dataset不同数据集的数量
        #数据集的位置不是用xy坐标来表示，每个图代表了一个数，从1开始
        #所以 当前的位置就是 当前对应行数*数据集的个数总和 + 当前对应列数 +1 因为subplot下标从1 开始
        f.add_subplot(len(models),len(data),inx*len(data)+i+1)
        plt.title(clt_name)

        # 输入数据是X，和聚类结果，轮廓系数要求有两个或两个以上的分类，但是上述的model第一个模型是NONE即没有分类 所以会报错
        # 加入try 和 pass  如果有聚类就计算其轮廓系数
        try:
            #这里的i是数据模型中的不同数组，clt_name是不同聚类方法 通过找i下标可以看出打印结果中不同的数据使用哪个模型评价效果好

            print(clt_name, i, silhouette_score(X, clt_res))
        except:
            pass

        #散点图  遍历所有的点 2维向量 X[p,0]相当横轴 x[P,1]相当于y轴
        [plt.scatter(X[p,0],X[p,1],color=colors[clt_res[p]])for p in range(len(X))]

plt.show()