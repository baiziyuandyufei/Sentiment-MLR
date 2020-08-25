import theano
import theano.tensor as T
from theano import shared
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
import argparse
import logging
import time
import collections
from WordLoader import WordLoader
import scipy.io


class RLstm(object):
    def __init__(self, words, grained, argv):
        parser = argparse.ArgumentParser()
        # 任务名称
        parser.add_argument('--name', type=str, default='test')
        # 随机数种子
        parser.add_argument('--rseed', type=int, default=int(1000 * time.time()) % 19921229)
        # LSTM隐藏层维度
        parser.add_argument('--dim_hidden', type=int, default=300)
        # 预训练词向量维度
        parser.add_argument('--dim_leaf', type=int, default=50)
        # softmax前加入丢弃层
        parser.add_argument('--dropout', type=int, default=1)
        # 正则化系数
        parser.add_argument('--regular', type=float, default=0.0001)
        # 词向量文件名
        parser.add_argument('--word_vector', type=str, default='wordvector/glove.refine.txt')
        # 解析传入的参数
        args, _ = parser.parse_known_args(argv)

        # 任务名称
        self.name = args.name
        logging.info('Model init: %s' % self.name)

        # 参数随机初始化种子
        self.srng = RandomStreams(seed=args.rseed)
        logging.info('RandomStream seed %d' % args.rseed)

        # 词向量维度
        self.dim_leaf = args.dim_leaf
        # 隐藏层维度
        self.dim_hidden = args.dim_hidden
        logging.info('dim: hidden=%s, leaf=%s' % (self.dim_hidden, self.dim_leaf))
        # 类别数量
        self.grained = grained
        logging.info('grained: %s' % self.grained)

        # 各类别词词典
        self.words = words
        # 各类别下词语数量
        self.num = {key: len(value) for key, value in words.items()}
        # 打印给类别词语的数量
        logging.info('vocabulary size: %s' % self.num)

        # 初始化神经网络
        self.init_param()
        # 加载词向量更新self.V中的参数
        self.load_word_vector('dataset/' + args.word_vector)
        # 编译模型函数
        self.init_function()

    # 初始化LSTM参数
    def init_param(self):
        # 定义服从均匀分布的共享变量参数
        def shared_matrix(dim, name, u=0, b=0):
            matrix = self.srng.uniform(dim, low=-u, high=u, dtype=theano.config.floatX) + b
            f = theano.function([], matrix)
            return theano.shared(f(), name=name)

        # LSTM隐藏层维度，词向量维度
        dh, dl = self.dim_leaf, self.dim_leaf
        # v_num存储各类别词语的数量, 定义self.V,存储各类别词向量矩阵
        v_num, self.V = [], []
        for key in ['negation', 'intensifier', 'sentiment', 'words']:
            v_num.append(self.num[key])
            self.V.append(shared_matrix((self.num[key], dl), 'V' + key, 0.01))
        # 否定词、强度词、情感词3个类别下的词语数量累加和[0, 否定词数量, 否定+强度词数量, 否定+强度+情感词数量]
        v_num = [sum(v_num[:i]) for i in range(len(self.num))]
        self.v_num = shared(np.array(v_num))
        # 把各类别的词向量矩阵按列合并为一个大的词向量矩阵
        self.V_all = T.concatenate(self.V, 0)

        # 5.2节中神经网络参数初始化，均匀分布参数
        u = lambda x: 1 / np.sqrt(x)

        # 输出层参数初始化
        self.W_hy = shared_matrix((dh, self.grained), 'W_hy', u(dh))
        self.b_hy = shared_matrix((self.grained,), 'b_hy', 0.)

        self.params = [self.W_hy, self.b_hy]

    def load_word_vector(self, fname):
        logging.info('loading word vectors...')
        loader = WordLoader()
        dic = loader.load_word_vector(fname)

        for v, key in zip(self.V, ['negation', 'intensifier', 'sentiment', 'words']):
            value = v.get_value()
            not_found = 0

            for words, index in self.words[key].items():
                word_list = eval(words)
                if (len(word_list) == 1) and (word_list[0] in dic.keys()):
                    value[index[1]] = list(dic[word_list[0]])
                else:
                    not_found += 1

            logging.info('word vector for %s, %d not found.' % (key, not_found))
            v.set_value(value)

    def init_function(self):
        logging.info('init function...')
        # 声明输入数据矩阵
        self.data = T.lmatrix()
        # 声明输出标记向量
        self.label = T.vector()

        # 句子向量=词向量和的平均
        x_t = T.mean(self.V_all[self.v_num[self.data[:, 0]]+self.data[:, 2]], 0)

        # 训练的预测表达式
        self.pred_for_train = T.nnet.softmax(T.dot(x_t, self.W_hy) + self.b_hy)[0]
        # 测试的预测表达式
        self.pred_for_test = T.nnet.softmax(T.dot(x_t, self.W_hy) + self.b_hy)[0]

        # 公式(4)中的第1项
        self.loss_supervised = -T.sum(self.label * T.log(self.pred_for_train))
        self.loss = self.loss_supervised

        # 计算各参数的梯度值
        gw = T.grad(self.loss, self.W_hy)
        gb = T.grad(self.loss, self.b_hy)

        # 编译训练函数
        logging.info("compiling func of train...")
        self.func_train = theano.function(
            inputs=[self.label, self.data],
            outputs=[self.loss, self.loss_supervised],
            updates=((self.W_hy, self.W_hy - 0.01 * gw),
                     (self.b_hy, self.b_hy - 0.01 * gb)),
            on_unused_input='warn')

        logging.info("compiling func of test...")
        self.func_test = theano.function(
            inputs=[self.label, self.data],
            outputs=[self.loss_supervised, self.pred_for_test],
            on_unused_input='warn')

    # 存储模型参数
    def dump(self, epoch):
        mdict = {}
        for param in self.params:
            val = param.get_value()
            mdict[param.name] = val
        scipy.io.savemat('mat/%s.%s' % (self.name, epoch), mdict=mdict)

    # 加载模型参数
    def load(self, fname):
        mdict = scipy.io.loadmat('mat/%s.mat' % fname)
        for param in self.params:
            if len(param.get_value().shape) == 1:
                pass
            if len(param.get_value().shape) >= 2:
                param.set_value(np.asarray(mdict[param.name], dtype=theano.config.floatX))
