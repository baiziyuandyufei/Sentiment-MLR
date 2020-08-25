import numpy as np
import theano
import argparse
import logging
import sys
from RLstm import RLstm as Model
from DataManager import DataManager
from Evaluator import EvaluatorList
import json


argv = sys.argv[1:]
parser = argparse.ArgumentParser()
# 任务名称
parser.add_argument('--name', type=str, default='test')
# 日志是写到终端显示还是写到文件
parser.add_argument('--screen', type=int, choices=[0, 1], default=1)
# 训练、验证和测试集文件所在路径
parser.add_argument('--dataset', type=str, default='mr/mr2')
# 一个批次内的样本数量
parser.add_argument('--batch_size', type=int, default=2000)
# 批次总数（不严格的周期数量）
parser.add_argument('--batch_num', type=int, default=200)
# 参数优化方法
parser.add_argument('--optimizer', type=str, default='ADAGRAD',
                    choices=['SGD', 'ADAGRAD', 'ADADELTA'])
# 参数学习速率
parser.add_argument('--lr', type=float, default=0.1)
# 词向量学习速率
parser.add_argument('--lr_vector', type=float, default=0.2)
# 每隔多少个批次输出一次准确率 设置为一个周期所含批次数量。
parser.add_argument('--interval', type=int, default=10)

# 解析设置的参数
args, _ = parser.parse_known_args(argv)

# 配置日志文件格式
logging.basicConfig(
        filename=('log/%s.log' % args.name) * (1-args.screen),
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(message)s',
        datefmt='%H:%M:%S')


# 加载语料文本，情感词，否定词，强度词文本
dm = DataManager(args.dataset, {'negation': 'negation.txt',
            'intensifier': 'intensifier.txt',
            'sentiment': 'sentiment.txt'})

# 从原始语料提取各类别词语
dm.gen_word_list()
# 将词语转成数值列表，构建出训练、验证和测试集
dm.gen_data()

# 构建模型
model = Model(dm.words, dm.grained, argv)
# 实例化评价器
Evaluator = EvaluatorList[dm.grained]


def do_train(label, data):
    loss = []
    for item_label, item_data in zip(label, data):
        item_data = item_data.astype(np.int64)
        item_loss = model.func_train(item_label, item_data)
        loss.append(item_loss)
    return np.sum(np.array(loss), 0) / len(loss)


def do_test(label, data):
    evaluator = Evaluator()
    loss = .0

    for item_label, item_data in zip(label, data):
        item_loss, item_pred = model.func_test(item_label, item_data)
        loss += item_loss
        evaluator.accumulate(item_label.reshape((1, -1)), item_pred.reshape(1, -1))
    logging.info('loss: %.4f' % (loss / len(data)))

    format_acc = lambda acc: ' '.join(['%s:%.4f' % (key, value) for key, value in acc.items()])
    acc = evaluator.statistic()
    logging.info('acc: %s' % format_acc(acc))

    return loss / len(data), acc


details = {'loss_train':[], 'loss_dev':[], 'loss_test':[],
        'acc_train':[], 'acc_dev':[], 'acc_test':[]}


for i in range(args.batch_num):
    # 获取一个批次的数据
    mini_batch = dm.get_mini_batch(args.batch_size)
    # 用一个批次的训练数据进行训练
    loss = do_train(*mini_batch)
    # 每个批次结束后打印损失
    format_loss = ' '.join(['%.3f' % x for x in loss. tolist()])
    logging.info('loss for batch %d: %s' % (i, format_loss))

    if (i+1) % args.interval == 0:
        now = {}
        # now['loss_train'], now['acc_train'] = do_test(*dm.data['train_small'])
        # now['loss_dev'], now['acc_dev'] = do_test(*dm.data['dev'])
        now['loss_test'], now['acc_test'] = do_test(*dm.data['test'])
        for key, value in now.items():
            details[key].append(value)
        with open('result/%s.txt' % args.name, 'w') as f:
            f.writelines(json.dumps(details))
        model.dump(i / args.interval)
