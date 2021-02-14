import numpy as np
import operator
from mnist import *


def knn_classify(test_data, train_dataset, train_label, k):
    train_dataset_amount = train_dataset.shape[0]  # 行数，也即训练样本的的个数，shape[1]为列数
    #print('train_label:', train_label)
    #print('train_dataset:', train_dataset)
    # 将输入test_data变成了和train_dataset行列数一样的矩阵
    test_rep_mat =np.tile(test_data, (train_dataset_amount, 1))  # tile(mat,(x,y)) Array类 mat 沿着行重复x次，列重复y次
    diff_mat = test_rep_mat - train_dataset
    #print('diff_mat:', diff_mat)
    # 求平方，为后面求距离准备
    sq_diff_mat = diff_mat ** 2
    #print('sq_diff_mat:', sq_diff_mat)
    # 将平方后的数据相加，sum(axis=1)是将一个矩阵的每一行向量内的数据相加，得到一个list，list的元素个数和行数一样;sum(axis=0)表示按照列向量相加
    sq_dist = sq_diff_mat.sum(axis=1)
    #print('sq_dist:', sq_dist)
    # 开平方，得到欧式距离
    distance = sq_dist ** 0.5
    #print('distance:', distance)

    # argsort 将元素从小到大排列，得到这个数组元素在distance中的index(索引)，dist_index元素内容是distance的索引
    dist_index = distance.argsort()
    #print('dist_index:', dist_index)

    class_count = {}
    for i in range(k):
        label = train_label[dist_index[i]]
        # 如果属于某个类，在该类的基础上加1，相当于增加其权重，如果不是某个类则新建字典的一个key并且等于1
        class_count[label] = class_count.get(label, 0) + 1
    # 降序排列
    class_count_list = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    #print('排序后的分类结果：', class_count_list)
    return class_count_list[0][0]


def get_cmd_pars(cmd_str):
    cmd_medum = []
    pars_ret = []
    type_ret = 'digit'
    cmd_list = cmd_str.split(sep=' ')  # 切割输入的字符串
    for cl in cmd_list:  # 这里将空串清除
        if cl != '':
            cmd_medum.append(cl)
    for cr in cmd_medum:  # 这里判断所有输入的参数是否是纯数字
        if not cr.isdigit():
            type_ret = 'string'
        else:
            pars_ret.append(int(cr))
    if len(pars_ret) < 2:  # 判断输入的参数是否大于2个数字
        type_ret = 'string'

    return type_ret, pars_ret


#################################################
if __name__ == '__main__':

    train_image_file = 'mnist\\train-images.idx3-ubyte'
    train_label_file = 'mnist\\train-labels.idx1-ubyte'
    test_image_file = 'mnist\\t10k-images.idx3-ubyte'
    test_label_file = 'mnist\\t10k-labels.idx1-ubyte'

    # 选择所有图片作为训练样本。
    #train_image_mat, train_label_list = read_image_label_all_vector(train_image_file, train_label_file)
    #	test_image_mat, test_label_list  = read_image_label_all_vector(test_image_file,test_label_file)
    # 选择部分数据作为训练集，第3个参数为偏移起始位置，第4个参数是训练样本数
    train_image_mat, train_label_list  = read_image_label_vector(train_image_file,train_label_file,0,60000)

    while True:
        # -----------交互式输入控制开始-----------------
        # 如果输入的样本数量为0，判断是否退出，如果不为0，继续开始分类。
        cmd = input('输入测试样本偏移和数量(比如 100 50):')
        type_ret, par_ret = get_cmd_pars(cmd)  # 解析输入的字符串
        if type_ret == 'digit':  # 如果全部为数字
            offset = par_ret[0]
            amount = par_ret[1]
            if amount == 0:
                continue
        else:  # 如果不是数字，提示是否退出程序
            cmd = input('格式不正确，输入Y(y)确定要退出:')
            if cmd == 'y' or cmd == 'Y':  # 输入了y则表示要退出程序
                break
            continue  # 没有输入y表示继续循环
        # -----------交互式输入控制结束-----------------

        # 根据前面的输入偏移和数量，开始读出测试样本
        test_image_mat, test_label_list = read_image_label_vector(test_image_file, test_label_file, offset, amount)

        # 开始分类
        err_count = 0.0  # 记录错误数量
        for i in range(len(test_image_mat)):
            print('当前进度：%2.2f%%' % (100.0 * i / len(test_image_mat)))
            # 利用knn算法进行分类
            class_result = knn_classify(test_image_mat[i], train_image_mat, train_label_list, 5)  # 计算分类结果
            #print("第 %d 张图片, 分类器结果: %d, 实际值: %d" % (i, class_result, test_label_list[i]), end=' ')
            # 判断分类结果是发和标签一致
            if (class_result != test_label_list[i]):
                #print(' 分类错误！', end=' ')
                err_count += 1.0
            # 打印错误率
            #print('当前错误率：%2.2f%%' % (100.0 * err_count / (i + 0.01)))

        print("\n总错误数: %d" % err_count)
        print("总错误率: %2.2f%%" % (100.0 * err_count / len(test_image_mat)))
