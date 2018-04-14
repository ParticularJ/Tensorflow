"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys

import tensorflow as tf

Py3 = sys.version_info[0] == 3

def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    if Py3:
      return f.read().replace("\n", "<eos>").split()
    else:
      return f.read().decode("utf-8").replace("\n", "<eos>").split()


def _build_vocab(filename):
  data = _read_words(filename)


  #统计所有字符出现的个数
  # 比如ababab 返回｛ａ:3, b:3｝
  counter = collections.Counter(data)
  # 通过比较，从词频高到低排序
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
  # 输出对应的单词,*可以理解为解压，　使得由上一步[('a',3),('b',3)]解压为
  # [(a,b),(3,3)]
  words, _ = list(zip(*count_pairs))
  # 使之每一个对应一个编号
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id


def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]


def ptb_raw_data(data_path=None):
  """Load PTB raw data from data directory "data_path".
  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.
  The PTB dataset comes from Tomas Mikolov's webpage:
  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.
  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_path = os.path.join(data_path, "ptb.train.txt")
  valid_path = os.path.join(data_path, "ptb.valid.txt")
  test_path = os.path.join(data_path, "ptb.test.txt")

  word_to_id = _build_vocab(train_path)
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  vocabulary = len(word_to_id)
  return train_data, valid_data, test_data, vocabulary


def ptb_producer(raw_data, batch_size, num_steps, name=None):
  """Iterate on the raw PTB data.
  This chunks up raw_data into batches of examples and returns Tensors that
  are drawn from these batches.
  Args:
    raw_data: one of the raw data outputs from ptb_raw_data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
    name: the name of this operation (optional).
  Returns:
    A pair of Tensors, each shaped [batch_size, num_steps]. The second element
    of the tuple is the same data time-shifted to the right by one.
  Raises:
    tf.errors.InvalidArgumentError: if batch_size or num_steps are too high.
  """
  with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
    # 将所有的ｗｏｒｄ_id 转化为ｔｅｎｓｏｒ形式
    raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
    # 计算总长度
    data_len = tf.size(raw_data)
    # 每个ｂａｔｃｈ的长度
    batch_len = data_len // batch_size
    # reahape [batch_size, batch_len]
    data = tf.reshape(raw_data[0: batch_size * batch_len],
                      [batch_size, batch_len])
    # epoch_size
    epoch_size = (batch_len - 1) // num_steps
    # 条件为假，打印ｍｅｓｓａｇｅ
    assertion = tf.assert_positive(
        epoch_size,
        message="epoch_size == 0, decrease batch_size or num_steps")
    # 依赖关系，必须assertion运行后，才会运行epoch_size
    with tf.control_dependencies([assertion]):
      # sameshape as epoch_size
      epoch_size = tf.identity(epoch_size, name="epoch_size")

    # 生成队列 ０ － epoch_size-1的整数。
    i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    # 取出[0-batch_size, num_steps]数据
    x = tf.strided_slice(data, [0, i * num_steps],
                         [batch_size, (i + 1) * num_steps])
    x.set_shape([batch_size, num_steps])
    # 取出[0-batch_size, num_steps]因为上一个y为下一个x的输入所以多一个
    y = tf.strided_slice(data, [0, i * num_steps + 1],
                         [batch_size, (i + 1) * num_steps + 1])
    y.set_shape([batch_size, num_steps])
    return x, y
