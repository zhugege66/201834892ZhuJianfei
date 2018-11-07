#计算某一类下的单词数目和该类下的单词总数
from os import listdir,path,makedirs
from math import log
import shutil
def compute_cate(cate_dir):
    count_word = {} #保存一类中单词word的次数
    cate_word = {}  #保存一类中单词出现的总数
    cate_list = listdir(cate_dir)
    for i in range(len(cate_list)):
        count_sum = 0 #一类中单词的总数
        doc_list = listdir(cate_dir + '/' + cate_list[i])
        for j in range(len(doc_list)):
            for line in open(cate_dir + '/' + cate_list[i] + '/' + doc_list[j]).readlines():
                word = line.strip('\n')
                count_sum += 1
                name = cate_list[i] + '_' + word
                count_word[name] = count_word.get(name,0)+1
        cate_word[cate_list[i]] = count_sum
        #print(cate_word.values())
    return cate_word,count_word

def computer_newProbablity(cate_list,cate_word,count_word,trainTotalWord,test_countword):
    train_word = cate_word[cate_list]  # 类train_list[k]下单词的总数
    pro = 0
    for m in range(len(test_countword)):
        name = cate_list + '_' + test_countword[m]
        if name in count_word.keys():
            test_wordNum = count_word[name]
        else:
            test_wordNum = 0.0
        new_pro = log((test_wordNum + 1) / (train_word + trainTotalWord))
        pro = pro + new_pro
    new_probablity = pro + log(train_word) - log(trainTotalWord)
    return new_probablity

def creat_testFile():
    f_train = open('train', 'w')
    f_test = open('test', 'w')
    cata_list = listdir('20news')
    for i in range(len(cata_list)):
        doc_list = listdir('20news' + '/' + cata_list[i])
        test_num = len(doc_list) * 0.2
        for j in range(len(doc_list)):
            if j < test_num:
                f = f_test
                if path.exists('test_dir') == False:
                    makedirs('test_dir')
                if path.exists('test_dir' + '/' + cata_list[i]) == False:
                    makedirs('test_dir' + '/' + cata_list[i])
                shutil.copyfile('20news' + '/' + cata_list[i] + '/' + doc_list[j],
                                'test_dir' + '/' + cata_list[i] + '/' + doc_list[j])
            else:
                f = f_train
                if path.exists('train_dir') == False:
                    makedirs('train_dir')
                if path.exists('train_dir' + '/' + cata_list[i]) == False:
                    makedirs('train_dir' + '/' + cata_list[i])
                shutil.copyfile('20news' + '/' + cata_list[i] + '/' + doc_list[j],
                                'train_dir' + '/' + cata_list[i] + '/' + doc_list[j])
            f.write('%s %s\n' % (doc_list[j],cata_list[i],))
    f_train.close()
    f_test.close()
    f.close()


def NBCprocess(train_dir,test_dir):
    f = open('classifiction','w')
    cate_word,count_word = compute_cate(train_dir)
    trainTotalWord = sum(cate_word.values()) #训练集中的单词总数
    test_list = listdir(test_dir)
    for i in range(len(test_list)):
        test_doclist = listdir(test_dir + '/' + test_list[i])
        for j in range(len(test_doclist)):
            test_countword = []
            for line in open(test_dir + '/' + test_list[i] + '/' + test_doclist[j]):
                word = line.strip('\n')
                test_countword.append(word)
            probablity = 0 #初始的概率
            train_list = listdir(train_dir)
            for k in range(len(train_list)):
                new_probablity = computer_newProbablity(train_list[k],cate_word,count_word,trainTotalWord,test_countword)
                if k == 0:
                    probablity = new_probablity
                    new_cate = train_list[k]
                    continue
                if new_probablity > probablity:
                    probablity = new_probablity
                    new_cate = train_list[k]
            f.write('%s %s\n' % (test_doclist[j],new_cate))
    f.close()

def computer_acc():
    class_result = {}
    cate = {}
    count = 0

    for line in open('classifiction').readlines():
        doc,cate_list = line.strip('\n').split(' ')
        class_result[doc] = cate_list
    for line in open('test').readlines():
        doc,cate_list = line.strip('\n').split(' ')
        cate[doc] = cate_list

    for key,values in cate.items():
        if class_result[key] == values:
            count += 1
    print('acc为:',count/len(cate))

if __name__ == '__main__':
    #creat_testFile()
    #NBCprocess('train_dir', 'test_dir')
    computer_acc()

