#encoding=utf-8
from numpy import *
import matplotlib
import matplotlib.pyplot as plt

def loaddateset(filename):
    with open(filename, 'r')as csvfile:
        dataset= [line.strip().split(', ') for line in csvfile.readlines()]     #读取文件中的每一行
        # dataset=[[int(i) if i.isdigit() else i for i in row] for row in dataset]    对于每一行中的每一个元素，将行列式数字化并且去除空白保证匹配的正确完成
        del (dataset[-1])
        cleanoutdata(dataset)#数据清洗
        dataset=precondition(dataset)#数据预处理
        dataset=[each[0:-2] for each in dataset]#去除最后一行数据，即类别
        return array(dataset)

def cleanoutdata(dataset):#数据清洗
    for row in dataset:
        for column in row:
            if column == '?'or column=='':#将所有包含？和空格的数据清洗掉
                dataset.remove(row)
                break
    for row in dataset:
        for column in row:
            if column == '?'or column=='':
                dataset.remove(row)
                break

def precondition(dataset):
    #将离散型数据换算成数值数据
    dict={'Private':0,'Self-emp-not-inc':1,'Self-emp-inc':2,'Federal-gov':3,
          'Local-gov':4,'State-gov':5,'Without-pay':6,'Never-worked':7,
          'Bachelors':0,'Some-college':1,'11th':2,'HS-grad':3,'Prof-school':4,
          'Assoc-acdm':5,'Assoc-voc':6,'9th':7,'7th-8th':8,'12th':9,'Masters':10,
          '1st-4th':11,'10th':12,'Doctorate':13,'5th-6th':14,'Preschool':15,
          'Married-civ-spouse':0, 'Divorced':1, 'Never-married':2,
          'Separated':3, 'Widowed':4, 'Married-spouse-absent':5,
          'Married-AF-spouse':6,'Tech-support':0, 'Craft-repair':1,
          'Other-service':2, 'Sales':3, 'Exec-managerial':4, 'Prof-specialty':5,
          'Handlers-cleaners':6, 'Machine-op-inspct':7, 'Adm-clerical':8,
          'Farming-fishing':9, 'Transport-moving':10, 'Priv-house-serv':11,
          'Protective-serv':12, 'Armed-Forces':13,'Wife':0, 'Own-child':1,
          'Husband':2, 'Not-in-family':3, 'Other-relative':4, 'Unmarried':5,
          'White':0, 'Asian-Pac-Islander':1, 'Amer-Indian-Eskimo':2, 'Other':3,
          'Black':4,'Female':0,'Male':1,'United-States':0, 'Cambodia':1,
          'England':2, 'Puerto-Rico':3, 'Canada':4, 'Germany':5,
          'Outlying-US(Guam-USVI-etc)':6, 'India':7, 'Japan':8, 'Greece':9,
          'South':10, 'China':11, 'Cuba':12, 'Iran':13, 'Honduras':14,
          'Philippines':15, 'Italy':16, 'Poland':17, 'Jamaica':18, 'Vietnam':19,
          'Mexico':20, 'Portugal':21, 'Ireland':22, 'France':23,
          'Dominican-Republic':24, 'Laos':25, 'Ecuador':26, 'Taiwan':27,
          'Haiti':28, 'Columbia':29, 'Hungary':30, 'Guatemala':31,
          'Nicaragua':32, 'Scotland':33, 'Thailand':34, 'Yugoslavia':35,
          'El-Salvador':36, 'Trinadad&Tobago':37, 'Peru':38, 'Hong':39,
          'Holand-Netherlands':40,'<=50K':'<=50K','>50K':'>50K','<=50K.':'<=50K','>50K.':'>50K'}
    dataset = [[int(column) if column.isdigit() else dict[column] for column in row] for row in dataset]#对于数据集中每一个元素，如果是离散性数据，转换为数值型
    return dataset

def distance(vecA,vecB):#计算欧氏距离
    return sqrt(sum(power(vecA-vecB,2)))

def randomcenter(dataset,k):#产生随机的k个质心
    datasetlen=shape(dataset)[1]
    center=mat(zeros((k,datasetlen)))
    for i in range(datasetlen):#对于dataset的每一个列，找出最小值和最大值，range代表最小值和最大值的差值，随机数可以设置为最小值加上差值*n,n小于1
        minnum=min(dataset[:,i])
        dis=float(max(dataset[:,i])-minnum)
        center[:,i]=minnum+dis*random.rand(k,1)
    return center

def kmeans(dataset,k,distanceMeature=distance,createCenter=randomcenter):#k_Means
    datasetlen=shape(dataset)[0]        #数据的条数
    cluster=mat(zeros((datasetlen,2)))  #datasetlen*2的矩阵用来保存当前行所对应的质心和到当前质心的距离
    center=createCenter(dataset,k)      #质心
    clusterChanged=True                #clusterChanged标志位用来描述质心是否发生改变
    while clusterChanged:
        clusterChanged=False
        for i in range(datasetlen):    #对于每一行的数据，设置一个最小距离和最小距离对应的索引，用来保存当前行与最近的质心的距离和质心的索引
            min_distance=Inf
            min_index=-1
            for j in range(k):
                dist=distanceMeature(center[j,:],dataset[i,:])  #计算第j个质心和第i行数据的距离，直接传递的是一行数据进行计算
                if dist<min_distance:                           #如果产生一个最小的距离，更新距离和质心索引，即将当前的行聚类到一个新的质心
                    min_distance=dist
                    min_index=j
            if cluster[i,0] != min_index:                       #如果所有点的质心不再发生改变，则结束聚类，否则继续聚类
                clusterChanged=True
            cluster[i,:]=min_index,dist**2                      #将上诉的最小距离和质心索引保存到cluster数组中
        #print(center)                                           结束聚类
        for cent in range(k):                                  #对于k个质心
            ptsInClust=dataset[nonzero(cluster[:,0].A == cent)[0]]  #取出跟当前循环到的第cent个质心相同的所有聚类到cent质心的距离的非零值
            center[cent,:]=mean(ptsInClust,axis=0)                  #压缩行，取得最小值的平均值
    return center,cluster

datasetname=r"C:\Users\yang\Desktop\adult.data"
mydate=loaddateset(datasetname)
center,cluster=kmeans(mydate,2)
with open(r"C:\Users\yang\Desktop\cluster.txt", 'w') as f:
    f.write(str(center.tolist()))
