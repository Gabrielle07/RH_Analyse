import pandas as pd
import numpy as np


#预处理数据：标准化和归一化
from sklearn.preprocessing import MinMaxScaler,StandardScaler

#特征数值化：标签和编码
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

#特征正规化
from sklearn.preprocessing import Normalizer

#LDA降维
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#PCA降维
from sklearn.decomposition import PCA

#以下属性可以放到一个列表中 用一个循环表示 遍历属性 用同一种处理方式

#sl:satifaction_level- false则表示用minmaxscaler处理 转换为0-1 True表示standard处理 转换成符合0,1下的分布
#le : last_evaluation
#npr:number project
#amh:average_monthly_hours -- False: MinMaxScaler, True : StandardScaler
#tsc : time_spend_company -- False : MinMaxScaler , True : standarscaler
#wa : work_accident-- False : MinMaxScaler , True : standarscaler

#以下属性放到一个列表中，属于数值化的处理，有标签化以及独热编码两种方式
#P15 : promotion_last_5years-- False : MinMaxScaler , True : standarscaler
#dp:departement ,True: labelincode,False


def hr_preprocessing(sl=False,le=False,npr=False,amh=False,tsc=False,wa=False,P15=False,dp=False,salary=False,low_d=False,ld_n=1):
    df=pd.read_csv('C:\\untitled2\\venv\\HR.csv')

    # 1 清洗数据 除去异常值，数据不算多不用进行抽样 若要得到标注的行 清洗数据应该在得到标注的后面
    # 去掉某些属性的空值
    df = df.dropna(subset=['satisfaction_level', 'last_evaluation'])

    # 去掉过大过小的数据 去掉相应的行
    df = df[df['satisfaction_level'] <= 1][df['salary'] != 'nme']

    # 2 得到标注
    label=df['left']
    #标注axis，以列删除 不处理label
    df=df.drop('left',axis=1)

    #3.特征选择 参考探索性数据分析的方块图 特征不多也可以不处理

    #4.特征处理 根据数据的特征选择处理方式 也可以不处理
    #将要处理的属性写上放在一个列表里，使用一个函数即可 不需再重写
    scaler_lst=[sl,le,npr,amh,tsc,wa,P15]
    column_lst=['satisfaction_level','last_evaluation','number_project','average_monthly_hours','time_spend_company','Work_accident','promotion_last_5years']

    #遍历sl是布尔型的数值 如果只要一列其实就循环了一次 做了一次判断而已 但如果要处理多个属性 则需要放入一个列表通过循环获取相应属性
    for i in range(len(scaler_lst)):
        #为假时
        if not scaler_lst[i]:

            #将dataframe里的值进行转换 用Minmax方法
            #转换为列的格式，然后再变回来 即reshape两次 二维向量，去第0个
            #column_lst[i]代表的是属性名字
            # 这里不需要reshape，输入的是整个列
            df[column_lst[i]]=MinMaxScaler().fit_transform(df[column_lst[i]].values.reshape(-1,1)).reshape(1,-1)[0]
        else:
            df[column_lst[i]]=StandardScaler().fit_transform(df[column_lst[i]].values.reshape(-1,1)).reshape(1,-1)[0]


    # 对于department进行数值化，可以进行label in coder 或者 onehotincoder
    scaler_list=[dp,salary]
    column_lst=['department','salary']

    for i in range(len(scaler_list)):

        #判断是否执行，scaler_list[i]是一个布尔值，参数由用户输入
        if not scaler_list[i]:

            #labelencoder不需要转换 但是labeleencoder是根据字母升序来编码的 所以要提前设置，利用字典的键值
            #按照字母编码是 low：1 medium： 2 high：0
            if column_lst[i]=='salary':

                #遍历字典选择值
                df[column_lst[i]]=[map_salary(s)for s in df['salary'].values]

            else: #对于别的属性可以直接进行label encoder
                df[column_lst[i]] = LabelEncoder().fit_transform(df[column_lst[i]])

            #再进行归一化处理
            df[column_lst[i]]=MinMaxScaler().fit_transform(df[column_lst[i]].values.reshape(-1,1)).reshape(1,-1)[0]

        else: #对于dataframe数据类型，直接用onehotencoder很复杂
            #用pandas里面的get_dummies处理，直接输入数据以及对应的列即可
            df=pd.get_dummies(df,columns=[column_lst[i]])


    #是否需要降维 先在主函数里指定参数 LDA降维 n_components 维数不可以大于类别数 由于均是0,1 降维后返回的都是一维
    #不采用LDA降维 PCA降维不受限制 可以不用使用标注 直接代入df直接fittransform 所以可以引入所有的类别
    if low_d:

        #返回label和降维 label即目标处理特征 返回查看 没有被处理
        df=pd.DataFrame(PCA(n_components=ld_n).fit_transform(df.values))

        return df,label

    #最终返回的特征和标注要应用于下一步的建模
    return df,label


#转换成字典
d= dict([('low',0),('medium',1),('high',2)])
def map_salary(s):
    #如果没有找到就默认返回0 低收入人群
    return d.get(s,0)



#建模函数
def hr_modeling(features,label):

    # train_test_split可以切分测试集和训练集
    from sklearn.model_selection import train_test_split

    #feares是个dataframe的结构，得到特征集的数值
    #用values可能会出错，因为经过降维的df变成了一个numpy的ndarray结构，这个结构没有values这个属性
    #其次对于数据切分，可以传入数组或者dataframe的结构，如果要保证dataframe这个结构，则在预处理hr_processing这个模型中，降维后的返回值应该再变成df结构
    f_v = features
    f_names=features.columns.values



    #得到标注值的数值
    l_v=label

    #先得到验证集和测试集的总和以及验证集 。 test_size 表示测试集占多少的比例，可以先取占比为0.2的验证集
    X_tt,X_validation,Y_tt,Y_validation=train_test_split(f_v,l_v,test_size=0.2)

    # 再区分测试集和训练集 验证集已经取出 剩下的验证集和测试集的综合为X_tt,Y_tt,其中的比例为3:1 ，测试集占0.25
    # test_size 针对第二个参数，train_size针对第一个参数的赞比
    X_train,X_test,Y_train,Y_test=train_test_split( X_tt,Y_tt,test_size=0.25)


    #KNN建模 ,NearestNeighors 可以得到最近的一些点
    from sklearn.neighbors import NearestNeighbors,KNeighborsClassifier

    # 朴素贝叶斯方法 特征必须是离散的 如果是0,1可用Bernoulli贝叶斯，GaussianNB默认特征为高斯分布
    from sklearn.naive_bayes import GaussianNB,BernoulliNB

    # 衡量指标
    from sklearn.metrics import accuracy_score, recall_score, f1_score

    # 决策树
    from sklearn.tree import DecisionTreeClassifier

    #随机森林
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import AdaBoostClassifier

    #画决策树，找到执行这个软件的目录
    import pydotplus
    from sklearn.tree import DecisionTreeClassifier,export_graphviz
    from six import StringIO

    #SVC支持向量机
    from sklearn.svm import SVC

    #逻辑斯特回归
    from sklearn.linear_model import LogisticRegression

    from sklearn.ensemble import GradientBoostingClassifier



    #人工神经网络
    #层级化容器
    from keras.models import Sequential

    #神经网络层，激活函数
    from keras.layers.core import Dense,Activation

    #随机梯度下降算法
    from keras.optimizers import SGD
    mdl=Sequential()

    #建立输入层,input_dim要和输入数据的维度保持一致
    #通常稠密层只要输入这一层要输出的数据维度，只有在输入层要输入这一层的输入维度以及输出的神经元个数

    mdl.add(Dense(50,input_dim=len(f_names)))
    mdl.add(Activation('sigmoid'))

    #对应的是输出层 不用管输入几个，上一层输出多少就会有多少输入。 只需要关注输出的维度
    mdl.add(Dense(2))
    #保证归一化
    mdl.add(Activation('softmax'))

    #指定梯度下降算法的参数 反向传播算法使用范围不广
    #对应输出层，不需要
    #优化器 设置学习率 即误差减小的梯度
    sgd=SGD(lr=0.01)

    #编译过程 增加损失函数，以及优化器 选择sgd
    mdl.compile(loss='mean_squared_error',optimizer='adam')

    #Y_tain必须是onehot形式，所以要进行转换 然后再转换成np形式,再选择梯度下降的迭代次数以及训练集数据的样本数
    #神经网络与之前的模型训练方式不同 所以单独拿出来训练
    #可以通过调节迭代次数以及升高梯度算法的学习率来提升评价指标 但是梯度优化器耗时过长
    #可以考虑更换优化器 如adam优化器，adam同时考虑梯度与动量，在最低点附近下降加快
    mdl.fit(X_train,np.array([[0,1] if i==1 else[1,0] for i in Y_train]),nb_epoch=1000,batch_size=2048)
    xy_lst = [(X_train, Y_train), (X_validation, Y_validation), (X_test, Y_test)]

    #画ROC曲线
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve,auc,roc_auc_score

    #初始化图形
    f=plt.figure()

    # 通过遍历，分别将训练集，验证集，测试集进行验证
    for i in range(len(xy_lst)):
        # 得到要验证的X
        X_part = xy_lst[i][0]
        # 得到对应的Y
        Y_part = xy_lst[i][1]
        # 如果用predict则输出的是连续值，若进行分类则应用predict_classes,输出的Y_pred也是onehot的形式
        #Y_pred = mdl.predict_classes(X_part)
        Y_pred=mdl.predict(X_part)
        # 准确率
        acc = accuracy_score(Y_part, Y_pred)
        # 召回率
        rec = recall_score(Y_part, Y_pred)
        # 综合反映
        f_score = f1_score(Y_part, Y_pred)

        #此时输出的Y_pred是与输出概率相关的矩阵值，但是画ROC曲线只需要得到其为正类的概率即可
        #要得到其为正类的概率，在上一步的fit中，进行onecode编码时，当为1时，第二列显示其数值即代表了判别为正类的概率
        #通过np将其转换为数组，取输出矩阵的第二列，再通过reshape转换为1行多列的数组
        Y_pred=np.array(Y_pred[:,1].reshpe(1,-1))[0]

        #圈定子图进行绘制
        f.add_subplot(1,3,i+1)
        #输入Y_ture和Y_score,在这里即真实的值（Y_Part是三个数据集真实的Y值)和预测值
        #返回值即可等到fpr,tpr以及阈值
        fpr,tpr,threshold = roc_curve(Y_part,Y_pred)

        #以fpr作为横坐标，tpr作为纵坐标进行绘图
        plt.plot(fpr,tpr)

        #打印AUC
        print('NN','AUC',auc(fpr,tpr))
        #填入真实值和预测值
        print('NN','AUC', roc_auc_score(Y_part,Y_pred))


        #print(i)
        #print('nn', 'ACC', acc)
        #print('nn', 'REC', rec)
        #print('nn', 'F_score', f_score)
    return


    #画图要用的包
    
    import os
    os.environ['PATH']+=os.pathsep+'C:\\Program Files\\Graphviz 2.44.1\\bin'



    # 将所有方法分类管理
    models = []

    #将对象加入到列表中 一个模型作为一个元组,元组的第一个元素是名字，第二是添加的模型对象

    models.append(('KNN', KNeighborsClassifier(n_neighbors=3)))
    models.append(('GaussianNB',GaussianNB()))
    models.append(('BernoulliNB',BernoulliNB()))
    
    #Gini系数的决策树
    models.append(('DecisionGini',DecisionTreeClassifier(min_impurity_split=0.1)))
    models.append(('DecisionTreeEntropy',DecisionTreeClassifier(criterion='entropy')))

    #SVM支持向量机
    models.append(('SVM Classifier',SVC(C=10000)))

    #随机森林 不设置的话，n_estimators 决策树数量默认为10，max features默认特征全取 划分时考虑所有的特征数，boostrap默认为True 即为有放回的采样
    #通过更改参数提升预测性能
    models.append(('RandomForest',RandomForestClassifier(n_estimators=81,max_features=None,bootstrap=False)))
    models.append(('RandomForest2',RandomForestClassifier()))

    #base_estimator即弱分类的分类器种类，默认为决策树。
    # algorithm有两种，SAMME SAMME.R，默认为SAMME.R，但若之前的base_estimator无法用概率判断，则要改为SAMME算法，如SVC
    models.append(('Adaboost',AdaBoostClassifier(base_estimator=SVC(),n_estimators=1000,algorithm='SAMME')))


    #线性回归分类器 通过调整参数提升模拟效果
    #penalty参数l1或者l2表示的是使用哪种方式正则化，tol是停止计算的精度
    #C越小表示正则化因子占得比例越大 solver即算法选择，默认liblinear，SAG:随机平均梯度下降算法,Max_iter是最大迭代次数 默认为100
    #本质依然是线性模型，如果调整参数依然没有太大的改变，则证明使用的数据集本身就不是线性可分的
    #其attribute有coef_参数，intercept_截距
    models.append(('LogiticRegression',LogisticRegression(C=1000,tol=1e-10,solver='sag',max_iter=10000)))


    #加入提升树模型,参数类似于决策树， learning_rate即衰减的百分比，默认的是mse算法
    #一般深度不会指定太深 一般深度比较小，经验值为6,n_examitor即树的数量
    models.append(('GBCT',GradientBoostingClassifier(max_depth=6,n_estimators=100)))


    #遍历列表 再拟合训练 过程一样 只是方法不同，可以放入一个列表中遍历 使用不同的方法
    for clf_name,clf in models:
        clf.fit(X_train,Y_train)
        #构造列表的元组分别存储训练集的xy，验证集xy，测试集的xy
        xy_lst=[(X_train,Y_train),(X_validation,Y_validation),(X_test,Y_test)]
        #通过遍历，分别将训练集，验证集，测试集进行验证
        for i in range(len(xy_lst)):
            #得到要验证的X
            X_part=xy_lst[i][0]
            #得到对应的Y
            Y_part=xy_lst[i][1]
            #通过循环 分别用以上训练集 验证集 以及测试集模拟模型结果
            Y_pred = clf.predict(X_part)
            #准确率
            acc = accuracy_score(Y_part, Y_pred)
            # 召回率
            rec = recall_score(Y_part, Y_pred)
            # 综合反映
            f_score = f1_score(Y_part, Y_pred)
            print(i)
            print(clf_name,'ACC',acc)
            print(clf_name, 'REC', rec)
            print(clf_name, 'F_score', f_score)

            #画决策树)

            #dot_data=StringIO() 再将下面的dot_data去掉 再out_file改成dot_data，最后graph输入的参数是dota.getvalue()
            dot_data=export_graphviz(clf,out_file=None,
                                     feature_names=f_names,
                                     class_names=['not left','left'],
                                     rounded=True,special_characters=True)
            graph=pydotplus.graph_from_dot_data(dot_data)
            #输出pdf文件 以gini系数划分
            graph.write_pdf('df_tree.pdf')


    print('validation')
    #用验证集进行验证
    Y_pred=knn_clf.predict(X_validation)

    #输入真实值和上一步由模型得到的预测值 预测值指标会有不同 因为每次随机切分的数据集不一样
    #准确率
    acc=accuracy_score(Y_validation,Y_pred)
    #召回率
    rec=recall_score(Y_validation,Y_pred)
    #综合反映
    f_score=f1_score(Y_validation,Y_pred)
    print(acc, rec, f_score)

    print('test')
    #再用测试集进行验证
    Y_pred=knn_clf.predict(X_train)
    # 准确率
    acc = accuracy_score(Y_train, Y_pred)
    # 召回率
    rec = recall_score(Y_train, Y_pred)
    # 综合反映
    f_score = f1_score(Y_train, Y_pred)
    print(acc,rec,f_score)

    print('trainning')
    #针对于训练集 计算其表现
    Y_pred=knn_clf.predict(X_test)
    # 准确率
    acc = accuracy_score(Y_test, Y_pred)
    # 召回率
    rec = recall_score(Y_test, Y_pred)
    # 综合反映
    f_score = f1_score(Y_test, Y_pred)
    print(acc, rec, f_score)
    
    #储存模型
    from sklearn.externals import joblib
    #填入此模型并命名 则会得到一个模型文件
    joblib.dump(knn_clf,'knn_clf')
    #使用这个模型
    knn_clf=joblib.load('knn_clf')



#线性回归函数 通过方块图 先找到和一些标注有关的特征 再应用线性回归模型
def regr_test(features,label):
    print('X',features)
    print('Y',label)

    #线性回归，岭回归，lasso回归
    from sklearn.linear_model import LinearRegression,Ridge,Lasso
   # regr=LinearRegression()
    #Alpha参数的大小会影响结果 不一定越大越好也不一定越小越好 要根据coef和MSE值调整
    regr=Ridge(alpha=0.95)
   # regr=Lasso(alpha=0.01)
    #训练拟合
    regr.fit(features.values,label.values)
    #预测值
    Y_pred=regr.predict(features.values)
    #查看参数
    print('Coef',regr.coef_)


    #模型评价：
    # 回归模型评估 不需要加入模型训练，是对所有的回归值和预测值进行评估
    # 对所有的回归模型进行评估
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    print('MSE', mean_absolute_error(label.values,Y_pred))
    print('MAE', mean_absolute_error(label.values, Y_pred))
    print('RSME',r2_score(label.values,Y_pred))

    #衡量回归好坏 引入平均平方误差
    from sklearn.metrics import mean_squared_error
    #输入预测值和标注的实际值比较
    print('MSE',mean_squared_error(Y_pred,label.values))

#在这里可以改参数
def main():
    #通过改变hr_preprocessing这个模型的参数，可以获得不同的数据集，进而影响评价指标,从而可以对比得到合适的数据处理方式
    features,label=hr_preprocessing(sl=False,le=False,npr=False,amh=False,tsc=True,wa=True,P15=True,dp=True,salary=True,low_d=False,ld_n=3)
    regr_test(features[['number_project','average_monthly_hours']],features['last_evaluation'])
   # hr_modeling(features,label)

