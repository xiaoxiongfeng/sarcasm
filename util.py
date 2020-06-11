import os
import json
import pandas as pd
import random
import string
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from flags import FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_num

class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
import re
def old_ans(input_file, type_):
    file = open(input_file)
    texts = []
    pre_texts = []
    labels = []
    ans = []
    for line in file:
        linedata = json.loads(line)
        if type_ == 'test':
            label = 'SARCASM'
        else:
            label = linedata['label']
        text = linedata['response']
        pre_text = [t.replace('\t', ' ') for t in linedata['context']]
        pre_text = '\t'.join(pre_text)
        texts.append(text)
        labels.append(label)
        pre_texts.append(pre_text)
        ans.append([label, text, pre_text])
    return ans
def get_beaty(s):
    s = str(s)
    s = re.sub("[%s]+" % (','), ' , ', s)
    s = re.sub("[%s]+" % ('!'), ' ! ', s)
    s = re.sub("[%s]+" % (':'), ' : ', s)
    s = re.sub("[%s]+" % ('.'), ' . ', s)
    s = re.sub("[%s]+" % ('?'), ' ? ', s)
    s = re.sub("[%s]+" % ('\''), ' \' ', s)
    s = re.sub('\(', ' ( ', s)
    s = re.sub('\)', ' ) ', s)
    s = re.sub('   ', ' ', s)
    s = re.sub('  ', ' ', s)
    return s

def get_pre():
    ans = []
    train_data = pd.read_csv("/home/xiaoxf/xxf/K-fold/data_external/pre/train_sarc_proc_bal.csv")
    for i in range(len(train_data)):

        # pa = string.punctuation
        #     # data = re.sub("[%s]+" % (pa), " ", sentence).split()

        pre_texts1 = train_data['Response 1'][i]
        pre_texts1 = get_beaty(pre_texts1)
        pre_texts2 = train_data['Response 2'][i]
        pre_texts2 = get_beaty(pre_texts2)
        texts = train_data['Ancestor'][i]
        texts = get_beaty(texts)
        label1 = train_data['Label 1'][i]
        label2 = train_data['Label 2'][i]
        if label1 == 1:
            label1 = 'SARCASM'
        else:
            label1 = 'NOT_SARCASM'
        if label2 == 1:
            label2 = 'SARCASM'
        else:
            label2 = 'NOT_SARCASM'
        ans.append([label1, texts, pre_texts1])
        ans.append([label2, texts, pre_texts2])
    return ans

def get_sarcasmdetection():
    ans = []
    file = open('/home/xiaoxf/xxf/K-fold/data_external/SarcasmDetection-master/dataset_csv.csv', encoding='utf-8')
    for line in file:
        if len(line.strip().split('\t')) == 2:
            text = line.strip().split('\t')[0]
            text = get_beaty(text)
            label = line.strip().split('\t')[1]
            if label == '0':
                label = 'NOT_SARCASM'
            else:
                label = 'SARCASM'
            ans.append([label, text, ''])
    file1 = open('/home/xiaoxf/xxf/K-fold/data_external/SarcasmDetection-master/sarcasm-dataset.txt', encoding='utf-8')
    for line in file1:
        text = line.strip()[:-1]
        text = get_beaty(text)
        label = line.strip()[-1]
        # print(label, text)
        if label == '0':
            label = 'NOT_SARCASM'
        elif label == '1':
            label = 'SARCASM'
        else:
            print(label)
        ans.append([label, text, ''])

    file2 = open('/home/xiaoxf/xxf/K-fold/data_external/SarcasmDetection-master/Tweet-Stream-API.txt', encoding='utf-8')
    for line in file2:
        text = line.strip()[:-1]
        text = get_beaty(text)
        label = line.strip()[-1]
        # print(label, text)
        if label == '0':
            label = 'NOT_SARCASM'
        elif label == '1':
            label = 'SARCASM'
        else:
            print(label)
        ans.append([label, text, ''])

    file3 = open('/home/xiaoxf/xxf/K-fold/data_external/SarcasmDetection-master/Tweets-after-Cleaning.txt', encoding='utf-8')
    for line in file3:
        text = line.strip()[:-1]
        text = get_beaty(text)
        label = line.strip()[-1]
        # print(label, text)
        if label == '0':
            label = 'NOT_SARCASM'
        elif label == '1':
            label = 'SARCASM'
        else:
            print(label)
        ans.append([label, text, ''])

    file4 = open('/home/xiaoxf/xxf/K-fold/data_external/SarcasmDetection-master/Tweets-with-no-label.txt', encoding='utf-8')
    for line in file4:
        if len(line) > 1:
            text = line.strip()[:-1]
            text = get_beaty(text)
            label = line.strip()[-1]
            # print(label, text)
            if label == '0':
                label = 'NOT_SARCASM'
            elif label == '1':
                label = 'SARCASM'
            else:
                print(label)
            ans.append([label, text, ''])
    return ans

def get_reddit_training():
    ans = []
    file = pd.read_csv('/home/xiaoxf/xxf/K-fold/data_external/reddit/reddit_training.csv')
    for i in range(len(file)):
        text = file['body'][i]
        text = get_beaty(text)
        label = file['sarcasm_tag'][i]
        if label == 'no':
            label = 'NOT_SARCASM'
        else:
            label = 'SARCASM'
        ans.append([label, text, ''])

    # file1 = pd.read_csv('/home/xiaoxf/xxf/K-fold/data_external/reddit/reddit_test.csv')
    # for i in range(len(file1)):
    #     text = file1['body'][i]
    #     text = get_beaty(text)
    #     label = file1['sarcasm_tag'][i]
    #     if label == 'no':
    #         label = 'NOT_SARCASM'
    #     else:
    #         label = 'SARCASM'
    #     ans.append([label, text, ''])
    return ans

def get_reddit_samps():
    ans = []
    # file = pd.read_csv('/home/xiaoxf/xxf/K-fold/data_external/reddit/train_10000samps.csv')
    file = pd.read_csv('/home/xiaoxf/xxf/K-fold/data_external/reddit/train-balanced-sarcasm.csv')
    for i in range(len(file)):
        text = str(file['comment'][i])
        text = get_beaty(text)
        pre_text = str(file['parent_comment'][i])
        pre_text = get_beaty(pre_text)
        label = file['label'][i]
        if label == 0:
            label = 'NOT_SARCASM'
        elif label == 1:
            label = 'SARCASM'
        ans.append([label, text, pre_text])
    return ans

def get_train_sarcasm():
    file = pd.read_csv('/home/xiaoxf/xxf/K-fold/data_external/twitter/data_train_sarcasm.csv')
    ans = []
    for i in range(len(file)):
        text = str(file['comment'][i])
        text = get_beaty(text)
        label = file['label'][i]
        if label == 0:
            label = 'NOT_SARCASM'
            # print(1111)
        else:
            label = 'SARCASM'
            # print(2233)
        if len(str(text).split()) > 5:
            ans.append([label, text, " "])

    file = pd.read_csv('/home/xiaoxf/xxf/K-fold/data_external/twitter/data_test_sarcasm.csv')
    for i in range(len(file)):
        text = str(file['comment'][i])
        text = get_beaty(text)
        label = file['label'][i]
        if label == 0:
            label = 'NOT_SARCASM'
            # print(1111)
        else:
            label = 'SARCASM'
            # print(2233)
        if len(str(text).split()) > 5:
            ans.append([label, text, " "])
    return ans

def get_headline():
    file = open('/home/xiaoxf/xxf/K-fold/data_external/twitter/Sarcasm_Headlines_Dataset_v2.json', 'r', encoding='utf-8')
    ans = []
    for line in file.readlines():
        dic = json.loads(line)
        text = dic['headline']
        text = get_beaty(text)
        if dic["is_sarcastic"] == 1:
            label = 'SARCASM'
        else:
            label = 'SARCASM'
        ans.append([label, text, " "])
    return ans

def get_sarcasm_dataset():
    file = pd.read_csv("/home/xiaoxf/xxf/K-fold/data_external/twitter/sarcasmania-dataset.csv")
    ans = []
    for i in range(len(file)):
        text = file['tweet'][i]
        text = get_beaty(text)
        label = file['sarcasm'][i]
        if label == '0':
            label = 'NOT_SARCASM'
        else:
            label = 'SARCASM'
        ans.append([label, text, " "])
    return ans

def get_pachong():
    file = pd.read_csv("/home/xiaoxf/xxf/K-fold/data_external/twitter/twitter_api.csv")
    ans = []
    for i in range(len(file)):
        text = file['text'][i]
        text = get_beaty(text)
        label = file['label'][i]
        ans.append([label, text, " "])
    return ans

class DataProcessor(object):
    def get_train_examples(self, data_dir):
        raise NotImplementedError()

    def get_dev_reddit_examples(self, data_dir):
        raise NotImplementedError()

    def get_dev_twitter_examples(self, data_dir):
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        raise NotImplementedError()

    def get_twitter_examples(self, data_dir):
        raise NotImplementedError()

    def get_reddit_examples(self, data_dir):
        raise NotImplementedError()


    def get_labels(self):
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file, type_):
        ans = old_ans(input_file, type_)
        if type_ == 'train': #add external data
            ans += 14 * ans
            ans += get_pre()
            ans += get_sarcasmdetection()
            ans += get_train_sarcasm()
            ans += get_sarcasm_dataset()
            ans += get_headline()
            ans += get_pachong()
            random.shuffle(ans)
        return ans

class MultiLabelTextProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        filename = FLAGS.train_file
        read_data = self._read_data(os.path.join(data_dir, filename), "train")
        return self._create_examples(read_data, "train")

    def get_dev_reddit_examples(self, data_dir):
        filename = FLAGS.dev_reddit_file
        read_data = self._read_data(os.path.join(data_dir, filename), "dev")
        return self._create_examples(read_data, "dev")

    def get_dev_twitter_examples(self, data_dir):
        filename = FLAGS.dev_twitter_file
        read_data = self._read_data(os.path.join(data_dir, filename), "dev")
        return self._create_examples(read_data, "dev")

    def get_twitter_examples(self, data_dir):
        filename = 'test_data/twitter_test/twitter_test.jsonl'
        read_data = self._read_data(os.path.join(data_dir, filename), "test")
        return self._create_examples(read_data, "test")

    def get_reddit_examples(self, data_dir):
        filename = 'test_data/reddit_test/reddit_test.jsonl'
        read_data = self._read_data(os.path.join(data_dir, filename), "test")
        return self._create_examples(read_data, "test")

    def get_labels(self):
        return list(pd.read_csv(os.path.join("/home/xiaoxf/xxf/K-fold/dataset/classes1.txt"), header=None)[0].values)

    def _create_examples(self, read_data, set_type):
        examples = []
        for (i, row) in enumerate(read_data):
            guid = "%s" % (i + 1)
            text_a = row[1]
            context_a = row[2]
            if set_type == 'test':
                labels = [0, 0]
            else:
                label = row[0]
                labels = []
                label_list = self.get_labels()
                for i in range(len(label_list)):
                    if label == label_list[i]:
                        labels.append(1)
                    else:
                        labels.append(0)
                if sum(labels) == 0:
                    print("ffffffffffffffffffffffffff")
                    break
            examples.append(InputExample(guid=guid, text_a=text_a, text_b = context_a, label=labels))
        return examples