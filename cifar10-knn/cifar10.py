import pickle
import operator
import numpy as np
import pandas as pd


# 数据获取
def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict


class KNearestNeighbor:
    def __init__(self):
        pass

    def train(self, X, y):
        '''X is size of N x D matrix, Y is 1-dimesion of size N'''
        self.Xtr = X
        self.ytr = y

    def predict(self, X, k):
        '''X is N x D matrix where each row is an example we wish to predict label for.
           k is the nearest neighbor algorithm'''
        num = X.shape[0]
        Ypred = np.zeros(num)
        for i in range(num):
            # 利用欧式距离
            distance = np.sum((self.Xtr - X[i, :]) ** 2, axis=1) ** 0.5
            #distance=np.sum((abs(self.Xtr-X[i,:])),axis=1)
            #distance=np.max(abs(self.Xtr-X[i,:]),axis=1)
            '''
            cos_mat1=np.sum((self.Xtr*X[i,:]),axis=1)
            cos_mat2=(np.sum((self.Xtr)**2,axis=1)*np.sum(X[i,:]**2))**0.5
            distance = cos_mat1 / cos_mat2
            '''
            # 对距离结果排序，得到从小到大索引
            sortedDistanceIndexs = distance.argsort()
            # k近邻的k循环,统计前k个距离最小的样本
            countDict = {}
            for j in range(k):
                countY = self.ytr[sortedDistanceIndexs[j]]  # 得到前k个从小到大索引的样本类别
                countDict[countY] = countDict.get(countY, 0) + 1  # 统计出现不存在则为0

            # 对前k个距离最小做value排序,找出统计次数最多的类别，作为预测类别
            sortedCountDict = sorted(countDict.items(), key=operator.itemgetter(1), reverse=True)
            Ypred[i] = sortedCountDict[0][0]
        return Ypred

# KNN对图像集做分类，计算准确率
top_num = int(input("输入验证集数据个数"))
train_data = unpickle('cifar-10-batches-py/data_batch_5')#得到字典
test_data = unpickle('cifar-10-batches-py/test_batch')

knn = KNearestNeighbor()
knn.train(train_data[b'data'], np.array(train_data[b'labels']))
#注意字典的索引是字节型的，比如要读取data,那么应该是dic[b'data']，字符串前面加b 才是字节
Ypred = knn.predict(test_data[b'data'][:top_num, :], 3)
accur = np.sum(np.array(Ypred) == np.array(test_data[b'labels'][:top_num])) / len(Ypred)
print("准确率 %f"%(accur))