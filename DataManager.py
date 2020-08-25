import numpy as np
import theano
import codecs
import random
random.seed(1229)


class DataManager(object):
    def __init__(self, data, words):
        def load_data(fname):
            data = []
            with open(fname) as f:
                for line in f:
                    # 以空白分隔
                    now = line.strip().split()
                    data.append((int(now[0]), now[1:]))
            return data

        self.origin_data = {}
        # 加载训练集、验证集、测试集
        for fname in ['train', 'dev', 'test']:
            self.origin_data[fname] = load_data('dataset/%s/%s.txt' % (data, fname))
        self.origin_words = {}
        # 读取情感词、否定词、强度词表
        for key, fname in words.items():
            self.origin_words[key] = load_data('dataset/wordlist/%s' % fname)

    # 从原始语料提取各类别词语
    def gen_word_list(self):
        # 将否定词词典{词语: 分值}、强度副词词典{词语: 分值}、情感词词典{词语: 分值}都存入self.words相应键下
        self.words = {}
        for key in ['negation', 'intensifier', 'sentiment']:
            words = {}
            for label, text in self.origin_words[key]:
                if repr(text) not in words.keys():
                    words[repr(text)] = int(label)
            self.words[key] = words

        # 将训练集、验证集、测试集中所有的词语构成的词典{词语: 0}存入self.words的键'words'下
        words = {}
        for key in ['train', 'dev', 'test']:
            for label, sent in self.origin_data[key]:
                for word in sent:
                    if repr([word]) not in words.keys():
                        words[repr([word])] = 0
        self.words['words'] = words

        # 返回self.words
        return self.words

    # 将词语转成数值列表，构建出训练、验证和测试集
    def gen_data(self):
        self.real_words = {'negation': {}, 'intensifier': {}, 'sentiment': {}, 'words': {}}

        def match(sent, l=3):
            for case, key in enumerate(['negation', 'intensifier', 'sentiment', 'words']):
                for length in reversed(range(1, l+1)):
                    now = repr(sent[:length])
                    if now in self.words[key].keys():
                        if now in self.real_words[key].keys():
                            subcase, index = self.real_words[key][now]
                            return [case, subcase, index], length
                        else:
                            subcase = self.words[key][now]
                            index = len(self.real_words[key])
                            self.real_words[key][now] = (subcase, index)
                            return [case, subcase, index], length

        # self.grained = 最大类别标记值+1
        self.grained = 1 + max([x for data in self.origin_data.values() for x, y in data])

        # self.data结构{数据种类: ([类别1, 类别2, ...], [实例1, 实例2, ])}
        # 类别i 用one-hot表示法
        # 实例i [词语1, 词语2, ...]
        # 词语i (词类别号, 词类别下词语的分值, 词类别下的词语索引)
        self.data = {}
        for key in ['train', 'dev', 'test']:
            data = []
            label = []
            for rating, sent in self.origin_data[key]:
                # 构建单词数字表示的一个文本
                result = []
                while len(sent) > 0:
                    res, length = match(sent)
                    result.append(res)
                    sent = sent[length:]
                if len(result) == 0:
                    continue
                data.append(np.array(result))
                # 构建类别one-hot表示，类别编号必须连续且大于0
                rat = np.zeros((self.grained), dtype=theano.config.floatX)
                rat[rating] = 1
                label.append(rat)
            self.data[key] = (label, data)

        # 小训练集 前10个类别，前10个训练集实例 注意：训练集的前10个实例必须有正例也有负例
        self.data['train_small'] = self.data['train'][0][::10], self.data['train'][1][::10], 

        # self.words结构为 {词类别: {词形: (对应于类别的分值: 在词类别下的索引号)}}
        self.words = self.real_words
        # 训练实例的索引
        self.index = list(range(len(self.data['train'][0])))
        # 当前训练实例索引号，如果该值大于等于len(self.index)，则打乱数据顺序，否则数据已经打乱顺序
        self.index_now = 0

        # 返回self.data
        return self.data

    def get_mini_batch(self, mini_batch_size=25):
        if self.index_now >= len(self.index):
            # random.shuffle(self.index)
            self.index_now = 0
        # 当前批次数据的起始索引，结束索引
        st, ed = self.index_now, self.index_now + mini_batch_size
        # 取出label序列
        label = np.take(self.data['train'][0], self.index[st:ed], 0)
        # 取出data序列
        data = np.take(self.data['train'][1], self.index[st:ed], 0)
        # 更新当前索引位置
        self.index_now += mini_batch_size
        # 返回label序列和data序列
        return label, data

    # 语料下各类别词语统计 论文中的表格1
    def analysis(self):
        num_sent = np.array([0, 0, 0, 0])
        num_word = np.array([0, 0, 0, 0])
        n_sent = 0
        for label, data in self.data.values():
            # 更新句子总数量
            n_sent += len(data)
            for sent in data:
                sent = np.array([[3, 0, 0], [3, 0, 1], [3, 0, 2], [3, 0, 3]])
                # 统计句子中各词类下词语数量
                now = np.bincount(sent[:, 0], minlength=4)
                # 更新含各类别词语的句子总数量
                num_sent += np.min([now, np.ones_like(now)], 0)
                # 更新各类别词语总数量
                num_word += now

        # 返回 含各类别词语的句子比例(表格1的统计), 各类别词语的比例
        return num_sent[:3] / n_sent, num_word[:3] / n_sent

    def gen_analysis_data(self, fname):
        def match(sent, l=3):
            for case, key in enumerate(['negation', 'intensifier', 'sentiment', 'words']):
                for length in reversed(range(1, l+1)):
                    now = repr(sent[:length])
                    if now in self.words[key].keys():
                        if now in self.real_words[key].keys():
                            subcase, index = self.real_words[key][now]
                            return [case, subcase, index], length
                        else:
                            print('error')
        with codecs.open(fname, "r", encoding='utf-8', errors='ignore') as fdata:
            origin_data = fdata.readlines()
            data = []
            for line in origin_data:
                sent = line.strip().split()
                result = []
                while len(sent) > 0:
                    res, length = match(sent)
                    result.append(res)
                    sent = sent[length:]
                data.append(np.array(result))
        return origin_data, data


if __name__ == '__main__':
    data = DataManager('sst', {'negation': 'negation.txt', \
            'intensifier': 'intensifier.txt', \
            'sentiment': 'sentiment.txt'})
    data.gen_word_list()
    data.gen_data()
    negation = [[] for i in range(32)]
    intensity = [[] for i in range(30)]

    def form(s):
        t = []
        for i in s:
            if (i == '.') or (i == ','):
                break
            else:
                t.append(i)
        return t

    for key in ['train', 'dev', 'test']:
        sents = data.data[key][1]
        origins = data.origin_data[key]
        for sent, origin in zip(sents, origins):
            for i, line in enumerate(sent.tolist()):
                if line[0] == 0:
                    word = line[2]
                    negation[word].append(origin[1][i:])
                if line[0] == 1:
                    word = line[2]
                    intensity[word].append(origin[1][i:])

    for key, value in data.words['intensifier'].items():
        name = '_'.join(eval(key))
        with open('dataset/analysis/intensifier/%s.txt' % name, 'w') as f:
            for sent in intensity[value[1]]:
                s = form(sent)
                if len(s) > 1:
                    f.writelines(' '.join(s)+'\n')
