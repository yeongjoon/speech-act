from konlpy.tag import Komoran

komoran = Komoran()

import re
def making_word_with_cate(word_2dlist, target_1dlist):
    i=-1
    word_with_cate=[]
    for word_1dlist in word_2dlist:
        i+=1
        word_with_cate.append([a+'_'+b+'_'+target_1dlist[i] for (a,b) in word_1dlist])

    return word_with_cate

def making_all_dict(dict_set):
    dict={}
    idx=0

    for i in dict_set:
        dict[i] = idx
        idx+=1

    return dict
def making_pre_target_list(list,dict):
    new_target_list = []

    for i in range(1,len(list)):
        new_target_list.append(dict[list[i - 1]])

    return new_target_list
def making_target_list(list, dict):
    target_list = []
    for target in list:
        target_list.append(dict[target])
    return target_list
def making_doc_dict(doc_list,dict):
    dict_list=[]
    dict_list_row =[0]*len(dict)
    for doc_word_list in doc_list:
        for j in doc_word_list:
            dict_list_row[dict[j]]+=1
        dict_list.append(dict_list_row)
        dict_list_row = [0]*len(dict)

    return dict_list
def making_doc_input(doc_list,dict):
    input_list=[]

    for list_ in doc_list:
        string = ''
        for (a,b) in list_:
            string+=a+'_'+b+' '
        input_list.append(string)

    return input_list

def making_word_dict(doc_list,dict):
    dict_list=[]
    dict_list_row =[0]*len(dict)
    for doc_word_list in doc_list:
        for j in doc_word_list:
            dict_list_row[dict[j]]+=1
        dict_list.append(dict_list_row)
        dict_list_row = [0]*len(dict)

    return dict_list

def load_data(filename):
    data = list(open(filename,'r').readlines())
    data = [s.strip() for s in data]
    return data

def morph_data(str):
    k = komoran.pos(str)
    s=""
    for item in k:
        s+=item[0].replace(" ",'')+item[1]+' '
    s = s[:-1]
    return s

def remain_morph(list):

    tmp = []
    for (a,b) in list:

        if 'SF' in b or 'EF' in b:
            tmp.append((a,b))
        if 'JKV' in b: # JKS/JKO->o   JKB->x JKV는 부를 때
            tmp.append((a, b))
        if 'EP' in b : #EF:어요 ETM/ETN은 좀 더 고려해보자
            tmp.append((a, b))
        if 'XSV' in b: #XSA도 고려해볼 것
            tmp.append((a, b))
        if 'JK' in b and 'JKC' not in b and 'JKG' not in b \
                and 'JKB' not in b and 'JKQ' not in b: # 을/를/이/가/(은/는 : JK) SVO
            tmp.append((a, b))
        if 'MAG' in b: #특히 좀_MAG일 때는 확실히 영향 미칠 것 같은데
            tmp.append((a, b))

    return tmp
def remain_morph2(list):

    tmp = []
    for (a, b) in list:
        if 'NNP' in b or 'NNG' in b:
            tmp.append((a, b))
        elif 'SF' in b or 'EF' in b:
            tmp.append((a, b))

        elif 'EP' in b : #EF:어요 ETM/ETN은 좀 더 고려해보자
            tmp.append((a, b))
        elif 'XSV' in b: #XSA도 고려해볼 것
            tmp.append((a, b))

        elif 'MAG' in b: #특히 좀_MAG일 때는 확실히 영향 미칠 것 같은데
            tmp.append((a, b))

    return tmp
def remain_morph3(list):

    tmp = []
    for (a, b) in list:
        if 'NNP' in b or 'NNG' in b:
            tmp.append((a, b))

    return tmp
def remain_morph5(list):

    tmp = []
    for (a,b) in list:

        if 'SF' in b or 'EF' in b:
            tmp.append((a,b))

    return tmp
def remain_morph6(list):

    tmp = []
    for (a,b) in list:
        if 'EF' in b:
            tmp.append((a,b))
        elif 'NNP' in b or 'NNG' in b:
            tmp.append((a, b))
    return tmp

def data_into_list(file_path):
    f = open(file_path, 'r')
    f.readline()
    data = [s.strip() for s in f.read().split('\n\n')]
    f.close() 
    return data


#한 object당 한 sentence
class Labeled_data():
    def __init__(self):
        self.speaker = None
        self.utterance = None
        self.prev_act = None
        self.cur_act = None
        self.i = None
        self.j = None

#paragraph 단위로 저장되어 있는 1d list를 sentence 기준의 2dlist로 변환
def organize_data(data):
    
    organized_data = []
    for i, paragraph in enumerate(data):
        sentence_list = []
        sentence_idx = 0
        for j, line in enumerate(paragraph.split('\n')):
            line = line.strip()
            if line is "":
                continue
            tmp = Labeled_data()
            tmp.speaker = line.split('\t')[0]
            tmp.utterance = line.split('\t')[1]
            tmp.cur_act = line.split('\t')[-1]
            tmp.i = i
            tmp.j = j
            if j == 0:
                tmp.prev_act = 'None'
            else:
                tmp.prev_act = sentence_list[sentence_idx-1].cur_act
            sentence_list.append(tmp)
            sentence_idx += 1
        organized_data.append(sentence_list)
    return organized_data

                
                




