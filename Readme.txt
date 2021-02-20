mnist数据集上的knn
代码中默认选取了全部的train_data作为训练集(60000）
输入:起点 偏移量（如 5000 200 表示从验证集的第5000个数据点开始选取200个数据）

cifar10数据集上的knn
代码中默认选取的第5个训练集中数据作为训练集
train_data = unpickle('cifar-10-batches-py/data_batch_5')#得到字典
top_num表示选取的验证集数据个数
输入top_num

具体可参考report中项目截图(上交时未同时提交mnist和cifar10原数据集)


