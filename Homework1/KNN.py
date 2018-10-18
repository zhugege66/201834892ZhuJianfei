from os import listdir
from math import log
import re
from numpy import *
from numpy import linalg
from operator import itemgetter

path1 = r'E:\code\git\Homework1\IDF'
path2 = r'E:\code\git\vsm'

def IDF():
    IDF_dict = {}
    IDF_map = {}
    cata_list = listdir(path2)
    for i in range(len(cata_list)):
        doc_list = listdir(path2 + '/' + cata_list[i])
        for j in range(len(doc_list)):
            for line in open(path2 + '/' + cata_list[i] + '/' + doc_list[j]).readlines():
                word = line.strip('\n')
                if word in IDF_map.keys():
                    IDF_map[word].add(doc_list[j])
                else:
                    IDF_map.setdefault(word,set())
                    IDF_map[word].add(doc_list[j])

    for word in IDF_map.keys():
        num = len(IDF_map[word])
        IDF = log(18828/num)/log(10)
        IDF_dict[word] = IDF
    f = open(path1 + '/ ' + 'IDF','w')
    for key,value in IDF_dict.items():
        f.write('%s %.6f\n' % (key,value))
    f.close()

def TF_IDF():
    IDF_dict = {}
    for line in open(r'E:\code\git\Homework1\IDF\idf').readlines():
        key = line.strip('\n').strip(' ')
        value = re.split(' ',key)
        word = value[0]
        idf = value[1]
        IDF_dict[word] = idf
    train_file = r'E:\code\git\Homework1\train_vector'
    test_file = r'E:\code\git\Homework1\test_vector'

    f_train = open(train_file + '/' + 'train','w')
    f_test = open(test_file + '/' + 'test','w')

    cata_list = listdir(path2)
    for i in range(len(cata_list)):
        doc_list = listdir(path2 + '/' + cata_list[i])
        test_num = len(doc_list) * 0.2
        for j in range(len(doc_list)):
            TF_map = {}
            totall_num = 0
            for line in open(path2 + '/' + cata_list[i] + '/' + doc_list[j]).readlines():
                totall_num += 1
                word = line.strip('\n')
                TF_map[word] = TF_map.get(word,0)+1
            if j < test_num:
                f = f_test
            else:
                f = f_train
            f.write('%s %s '%(cata_list[i],doc_list[j]))
            for key,value in TF_map.items():
                TF = float(value)/float(totall_num)
                f.write('%s %.6f ' %(key,TF * float(IDF_dict[key])))
            f.write('\n')
    f_train.close()
    f_test.close()
    f.close()


def Knn():
    train_list = r'E:\code\git\Homework1\train_vector\train'
    test_list = r'E:\code\git\Homework1\test_vector\test'
    knn_file = r'E:\code\git\Homework1\knn'

    train_map = {}

    for line in open(train_list).readlines():
        line_value = line.strip('\n').strip(' ')
        line_train = re.split(' ',line_value)
        train_word = {}
        num = len(line_train) - 1
        for i in range(2,num,2):
            train_word[line_train[i]] = line_train[i+1]
        temp_key = line_train[0] + '_' + line_train[1]
        train_map[temp_key] = train_word

    test_map = {}

    for line in open(test_list).readlines():
        line_value = line.strip('\n').strip(' ')
        line_test = re.split(' ',line_value)
        test_word = {}
        num = len(line_test) - 1
        for i in range(2, num, 2):
            test_word[line_test[i]] = line_test[i + 1]
        temp_key = line_test[0] + '_' + line_test[1]
        test_map[temp_key] = test_word
    count = 0
    right_count = 0
    f_knn = open(knn_file + '/' + 'knn','w')
    for key,value in test_map.items():
        classifyResult = knn_compute(key, value, train_map)
        count += 1
        classifyRight = key.split('_')[0]
        f_knn.write('%s %s\n' % (classifyRight, classifyResult))
        if classifyRight == classifyResult:
            right_count += 1
        print('统计%d的准确率为 %f' %(count,float(right_count) / float(count)))
        if count > 1000:
            f_knn.close()
            break

    #accuracy = float(right_count) / float(count)
    #print(accuracy)


def knn_compute(cate_Doc, testDic, trainMap):
    sim_map = {}
    for item in trainMap.items():
        similarity = computeSim(testDic, item[1])
        sim_map[item[0]] = similarity

    sortedSimMap = sorted(sim_map.items(), key=itemgetter(1), reverse=True)

    k = 20
    cateSimMap = {}
    for i in range(k):
        cate = sortedSimMap[i][0].split('_')[0]
        cateSimMap[cate] = cateSimMap.get(cate, 0) + sortedSimMap[i][1]

    sortedCateSimMap = sorted(cateSimMap.items(), key=itemgetter(1), reverse=True)

    return sortedCateSimMap[0][0]


def computeSim(testDic, trainDic):
    testList = []
    trainList = []

    for word, weight in testDic.items():
        if word in trainDic.keys():
            testList.append(float(weight))
            trainList.append(float(trainDic[word]))
    testVect = mat(testList)
    trainVect = mat(trainList)
    num = float(testVect * trainVect.T)
    denom = linalg.norm(testVect) * linalg.norm(trainVect)
    return float(num) / (1.0 + float(denom))


if __name__ == '__main__':
    #IDF()
    #TF_IDF()
    Knn()
