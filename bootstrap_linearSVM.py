import numpy as np
import data_preprocessing as pre
import re
#from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.svm.libsvm import *
from sklearn.model_selection import train_test_split
#from sklearn.multiclass import OneVsRestClassifier
#from sklearn.linear_model import LogisticRegression
import json
import random
import pickle
from tqdm import tqdm
#from svmutil import *

import sys

class Bootstrap():
    def __init__(self, unlabeled_data=None):
        #form of sentence, not changed into bag of words yet.

        self.Test_X = None              #dataset for calculating model accuracy
        self.Test_Y = None              #after one iteration

        self.baseTrain_X = None
        self.baseTrain_Y = None

        self.target_dict = None
        self.inv_target_dict = None

        self.morph_list = None
        self.target_list = None


        #It is sliced into N dataset.
        self.once_filtered_X = None             #ex: 1000 samples
        self.once_filtered_Y = None             #form of list of list

        self.twice_filtered_X = None            #ex: 300samples of something
        self.twice_filtered_Y = None

        self.first_filter_threshold = 1
        self.second_filter_threshold = 1
        self.number_of_bagging_model = 5

        self.unlabeled_data = None

    def yeongjoon_initialize(self):
        labeled_data = pre.data_into_list('./labeled_data/hotel_corpus.tsv')
        labeled_data.extend(pre.data_into_list('./labeled_data/schedule_corpus.tsv'))

        morph_list = []
        target_list = []
        organized_data = pre.organize_data(labeled_data)    #[paragraph_number][length of each paragraph]
        for paragraph in organized_data:
            for sentence in paragraph:
                morph_list.append(pre.morph_data(sentence.utterance))
                target_list.append(sentence.cur_act)

        #print(set(target_list))

        all_target_dict = pre.making_all_dict(list(set(target_list)))
        self.inv_target_dict = {v: k for k, v in all_target_dict.items()}
        all_target_1dlist = pre.making_target_list(target_list, all_target_dict)
        target_list = all_target_1dlist
        self.target_dict = pre.making_all_dict(list(set(target_list)))
        #self.inv_target_dict = {v: k for k, v in self.target_dict.items()}
        print(self.inv_target_dict)

        self.morph_list = morph_list
        self.target_list = target_list

        #Train_X, Test_X, Train_Y, Test_Y = train_test_split(morph_list, target_list, test_size=0.2, random_state=42)
        Train_X, Test_X, Train_Y, Test_Y = train_test_split(morph_list, target_list, test_size=0.1, shuffle=True, random_state=42)
        self.baseTrain_X = Train_X
        self.baseTrain_Y = Train_Y
        self.Test_X = Test_X
        self.Test_Y = Test_Y
        
    def small_data_score(self, number):
        Train_X, Test_X, Train_Y, Test_Y = train_test_split(self.morph_list, self.target_list, test_size=0.1, shuffle=False, random_state=42)
        score_sum = 0
        for i in range(number):
            quotient = len(Train_X) // number
            tmp_Train_X = Train_X[i*quotient:(i+1)*quotient]
            tmp_Train_Y = Train_Y[i*quotient:(i+1)*quotient]
            tmpModel = SVMModel(tmp_Train_X, tmp_Train_Y)

            word_dict, model = tmpModel.word_dictionary(), tmpModel.model()

            """ Testing Accuracy Here"""
            test_array = np.zeros((len(self.Test_X), len(word_dict)))
            for j ,v in enumerate(self.Test_X):
                for word in v.split():
                    if word in word_dict:
                        test_array[j][word_dict[word]] += 1
            score = model.score(test_array, self.Test_Y)
            score_sum += score
            print(i, "th iteration: ", score)
        print("Average score: %.4f" % (score_sum/number))


    def initialize_dataset(self):
        labeled_data = pre.load_data('./labeled_data/hotel_corpus.tsv')
        labeled_data.extend(pre.load_data('./labeled_data/schedule_corpus.tsv'))
        labeled_data_matrix = [re.split('[\t\n]', s) for s in labeled_data]
        target_list = []
        morph_list = []
        only_target_list = []
        for row in labeled_data_matrix:
            if len(row) <= 1:
                target_list.append('None')
                continue

            elif row[0] == 'Speaker':
                target_list.append('None')
                continue
            if len(row) == 3:
                print(row[1])
            elif len(row) == 4:
                target_list.append(row[-1])
                only_target_list.append(row[-1])
            morph_list.append(pre.morph_data(row[1]))

        #print((morph_list[:15])) 
        #all_target_dict = pre.making_all_dict(list(set(target_list)))      #target들의 dictionary
        #all_target_1dlist = pre.making_target_list(target_list, all_target_dict)
        all_target_dict = pre.making_all_dict(list(set(only_target_list)))      #target들의 dictionary
        all_target_1dlist = pre.making_target_list(only_target_list, all_target_dict)


        #none_idx = int(all_target_dict['None'])
        target_list = all_target_1dlist
        #while target_list.count(none_idx) != 0:
        #    target_list.remove(none_idx)

        self.target_dict = pre.making_all_dict(list(set(target_list)))
        #self.target_dict = pre.making_all_dict(list(set(only_target_list)))

        #print('len of target dict', len(self.target_dict))
        self.inv_target_dict = {v: k for k, v in self.target_dict.items()}
        #for key in self.target_dict:
        #    print (key)
        #print(len(target_list), len(morph_list))
        #print(morph_list[:10])
        Train_X, Test_X, Train_Y, Test_Y = train_test_split(morph_list, target_list, test_size=0.2, random_state=42)
        self.baseTrain_X = Train_X
        self.baseTrain_Y = Train_Y
        self.Test_X = Test_X
        self.Test_Y = Test_Y

    def train(self, unlabeled_data):

        Train_X, Train_Y = self.baseTrain_X, self.baseTrain_Y
        baseModel = SVMModel(Train_X, Train_Y)
        self.once_filtered_X , self.once_filtered_Y = self.first_filter(baseModel, unlabeled_data)
        #self.twice_filtered_X, self.twice_filtered_Y = self.ensemble(self.once_filtered_X, self.once_filtered_Y, unlabeled_data)
        self.twice_filtered_X, self.twice_filtered_Y = self.cluster(self.once_filtered_X, self.once_filtered_Y, unlabeled_data)
        self.baseTrain_X += self.twice_filtered_X
        self.baseTrain_Y += self.twice_filtered_Y
        print("length of current golden set: ",len(self.baseTrain_X), '\n')
        """
        1. train baseline model with (golden set(ex: 1000) + additional data)
        2. tag one batch of unlabeled data(ex: 2500)
        3. for top a% of unlabeled data that has higher score, split it into
           n datasets.(ex: 5)
        4. for each dataset, use (golden set + itself) as training data
        5. tag each bag model with whole unlabeled data(ex: 2500)
        6. Using ensemble, save top b% of unlabeled data
        """

    def cluster_with_prev_act(self, X_list, Y_list, unlabeled_data):
        pass

    def cluster_with_cur_act(self, X_list, Y_list, unlabeled_data):
        pass

    def first_filter(self, baseModel, unlabeled_data):
        word_dict, model = baseModel.word_dictionary(), baseModel.model()


        """ Testing Accuracy Here"""
        test_array = np.zeros((len(self.Test_X), len(word_dict)))
        for i ,v in enumerate(self.Test_X):
            for word in v.split():
                if word in word_dict:
                    test_array[i][word_dict[word]] += 1
        score = model.score(test_array, self.Test_Y)
        #_, (score,_,_), _ = svm_predict(self.Test_Y, test_array.tolist(), model, '-b 1')
        print("Accuracy with", len(self.baseTrain_X)," examples: ",score)

        array_form = np.zeros((len(unlabeled_data), len(word_dict)))
        for i, v in enumerate(unlabeled_data):
            for word in v.split():
                if word in word_dict:
                    array_form[i][word_dict[word]] += 1

        filtered_x_list = []
        filtered_y_list = []

        predicted_probs = model.decision_function(array_form)
        #tmp_label = [0 for i in range(len(unlabeled_data))]
        #_,_,predicted_probs = svm_predict(tmp_label, array_form.tolist(), model, '-b 1')

        for i, v in enumerate(predicted_probs):
            max_prob = -999999
            idxtmp=-1
            for j, prob in enumerate(v):
                if prob >= max_prob:
                    idxtmp = j
                    max_prob = prob
            #print(v[idxtmp])
            if idxtmp != -1 and v[idxtmp] >= self.first_filter_threshold:
                filtered_x_list.append(unlabeled_data[i])
                filtered_y_list.append(idxtmp)

        return filtered_x_list, filtered_y_list
        """
        return some unlabeled data, and label
        of which score is higher than first threshold
        in form of N list(N = number of bagging model)
        shape = [number of bagging model][number of sentences in each bagging model]

        """

    def ensemble(self, X_list, Y_list, unlabeled_data):
        quotient = len(X_list) // self.number_of_bagging_model   #for each bagging model, make model and ensemble
        ensemble_predict = np.zeros((len(unlabeled_data), len(self.target_dict)))
        #print(X_list, Y_list)
        #print(len(Y_list), "fuck you!")
        #print(len(set(Y_list)), "fuck fuck")
        print("additional ", len(X_list), "data in ensemble")
        for i in range(self.number_of_bagging_model):
            tmp_X_list = X_list[i*quotient:(i+1)*quotient]
            tmp_Y_list = Y_list[i*quotient:(i+1)*quotient]

            tmpModel = SVMModel(self.baseTrain_X+tmp_X_list, self.baseTrain_Y+tmp_Y_list)
            #print(self.baseTrain_Y[:20])
            #print(tmp_Y_list[:20])
            word_dict, model = tmpModel.word_dictionary(), tmpModel.model()
            array_form = np.zeros((len(unlabeled_data), len(word_dict)))
            for i, v in enumerate(unlabeled_data):
                for word in v.split():
                    if word in word_dict:
                        array_form[i][word_dict[word]] += 1

            tmp_probs = model.decision_function(array_form)
            #tmp_label = [0 for i in range(len(unlabeled_data))]
            #_,_,tmp_probs = svm_predict(tmp_label, array_form.tolist(), model, '-b 1')
            ensemble_predict += np.asarray(tmp_probs)

        ensemble_predict /= self.number_of_bagging_model
        twice_filtered_x_list = []
        twice_filtered_y_list = []
        for i, v in enumerate(ensemble_predict.tolist()):
            max_prob = -999999
            idxtmp = -1
            for j, prob in enumerate(v):
                if prob >= max_prob:
                    idxtmp = j
                    max_prob = prob
            if idxtmp != -1 and v[idxtmp] >= self.second_filter_threshold:
                twice_filtered_x_list.append(unlabeled_data[i])
                twice_filtered_y_list.append(idxtmp)

        return twice_filtered_x_list, twice_filtered_y_list

        """
        for each dataset in X_list, make a svm bagging model.
        ensemble the result score(tag unlabeled data)
        return some unlabeled data, and label of which
        score is higher than second threshold
        """

class SVMModel():
    def __init__(self, Train_X, Train_Y, Test_X=None, Test_Y=None):
        self.Train_X = Train_X
        self.Train_Y = Train_Y
        self.Test_X = Test_X
        self.Test_Y = Test_Y
        print("Initializing SVM model with ", len(self.Train_X), " data")

    #dictionary containing all word in training data
    def word_dictionary(self):
        word_dict = {}
        count=0
        for sentence in self.Train_X:
            for word in sentence.split():
                if word not in word_dict:
                    word_dict[word] = count
                    count += 1
        return word_dict

    def model(self):
        word_dict = self.word_dictionary()
        array_form = np.zeros((len(self.Train_X), len(word_dict))) #bag of words 형태의 문장 표현
        for i, v in enumerate(self.Train_X):
            for word in v.split():
                if word in word_dict:
                    array_form[i][word_dict[word]] += 1

        #clf = svm_train(self.Train_Y, array_form.tolist(), '-s 0 -t 0 -b 1 -h 0')
        
        clf = LinearSVC()
        clf.fit(array_form, self.Train_Y)

        #print(array_form.shape, "in model func")
        #print(clf.predict_proba(array_form).shape, "in model function")
        return clf

def read_json(file_path):
    with open(file_path) as f:
        json_data = json.load(f)
    unlabeled_data  = []
    for data in json_data["data"]:
        unlabeled_data.append(pre.morph_data(data["question_text"]))
        unlabeled_data.append(pre.morph_data(data["answer_text"]))

    return list(set(unlabeled_data))

if __name__ == '__main__':
    
    f = open('./unlabeled_data/revised_unlabeled_data', 'r')
    whole_unlabeled_data = []
    for line in f.readlines():
        whole_unlabeled_data.append(line)
    
    random.shuffle(whole_unlabeled_data) #shuffle
    batch_size = 5000
    cur_idx = 0

    
    boot = Bootstrap()
    boot.yeongjoon_initialize()
    boot.small_data_score(9)
    #sys.exit(1)
    #boot.initialize_dataset()
    tmp_Train_X = boot.baseTrain_X
    tmp_Train_Y = boot.baseTrain_Y

    quotient = len(boot.baseTrain_X) // 9
    for j in range(9):
        boot.yeongjoon_initialize()
        boot.baseTrain_X, boot.baseTrain_Y = tmp_Train_X[j*quotient:(j+1)*quotient], tmp_Train_Y[j*quotient:(j+1)*quotient]
        for i in range(len(whole_unlabeled_data) // batch_size):
            unlabeled_data = whole_unlabeled_data[int(i*batch_size): (i+1)*int(batch_size)]
            boot.train(unlabeled_data)
        print("Iteration End\n\n\n\n")
    """
    for i in range(len(whole_unlabeled_data) // batch_size):
            unlabeled_data = whole_unlabeled_data[int(i*batch_size): (i+1)*int(batch_size)]
            boot.train(unlabeled_data)
    """
    with open('./outputs/child_output_1.0', 'wb') as fo:
        pickle.dump(boot.baseTrain_X, fo)
        pickle.dump(boot.baseTrain_Y, fo)
    
    print("done!")
