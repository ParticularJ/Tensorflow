import glob
import os.path
import random
import numpy as np
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
import cv2
from tensorflow.python.platform import gfile



BOTTLENECK_TENSOR_SIZE = 2048
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'


MODEL_DIR = 'datasets/inception_dec_2015'
MODEL_FILE= 'tensorflow_inception_graph.pb'

CACHE_DIR = 'datasets/bottleneck'
INPUT_DATA = 'datasets/flower_photos'

INPUT_target_1 = 'datasets/target/fht_001.jpg'
INPUT_target_2 = 'datasets/target/fht_003.jpg'
INPUT_target_3 = 'datasets/target/hhl_004.jpg'
INPUT_target_4 = 'datasets/target/lft_001.jpg'
INPUT_target_5 = 'datasets/target/lft_004.jpg'
#INPUT_target_6 = 'datasets/target/14283011_3e7452c5b2_n.jpg'
#INPUT_target_7 = 'datasets/target/16041975_2f6c1596e5.jpg'
#INPUT_target_8 = 'datasets/target/126012913_edf771c564_n.jpg'
#INPUT_target_9 = 'datasets/target/141340262_ca2e576490.jpg'
#INPUT_target_10 = 'datasets/target/149782934_21adaf4a21.jpg'
VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10



LEARNING_RATE = 0.01
STEPS = 20000
BATCH = 10
image_labels=["黄鹤楼", "雷峰塔", "飞虹塔"]
# image_raw_data = plt.imread('datasets/flower_photos/daisy/5547758_eea9edfd54_n.jpg')

def create_image_lists(testing_percentage, validation_percentage):
    result = {}
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']

        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list: continue

        label_name = dir_name.lower()

        # 初始化
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)

            # 随机划分数据
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(base_name)
            elif chance < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)

        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
        }
    return result



def get_image_path(image_lists, image_dir, label_name, index, category):
    label_lists = image_lists[label_name]
    category_list = label_lists[category]
    mod_index = index % len(category_list)
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path



def get_bottleneck_path(image_lists, label_name, index, category):
    return get_image_path(image_lists, CACHE_DIR, label_name, index, category) + '.txt'


# inception处理一张图片，得出特征向量,获得image_data即可。
def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):

    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})

    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values

# 获取一张图片的张量


def get_tensor(sess, INPUT_ta, image_data_tensor, bottleneck_tensor):
    bottlenecks_ima = []
    image_data_1 = gfile.FastGFile(INPUT_ta, 'rb').read()
    image_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data_1})
    image_values_1 = np.squeeze(image_values)
    bottlenecks_ima.append(image_values_1)
    return bottlenecks_ima



def get_or_create_bottleneck(sess, image_lists, label_name, index, category, jpeg_data_tensor, bottleneck_tensor):
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(CACHE_DIR, sub_dir)
    if not os.path.exists(sub_dir_path): os.makedirs(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index, category)
    if not os.path.exists(bottleneck_path):

        image_path = get_image_path(image_lists, INPUT_DATA, label_name, index, category)\


        image_data = gfile.FastGFile(image_path, 'rb').read()

        bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)

        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:

        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]

    return bottleneck_values



def get_random_cached_bottlenecks(sess, n_classes, image_lists, how_many, category, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    for _ in range(how_many):
        label_index = random.randrange(n_classes)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(65536)
        bottleneck = get_or_create_bottleneck(
            sess, image_lists, label_name, image_index, category, jpeg_data_tensor, bottleneck_tensor)
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)

    return bottlenecks, ground_truths



def get_test_bottlenecks(sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    label_name_list = list(image_lists.keys())
    for label_index, label_name in enumerate(label_name_list):
        category = 'testing'
        for index, unused_base_name in enumerate(image_lists[label_name][category]):
            bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, index, category,jpeg_data_tensor, bottleneck_tensor)
            ground_truth = np.zeros(n_classes, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
    return bottlenecks, ground_truths


def main():
    image_lists = create_image_lists(TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
    n_classes = len(image_lists.keys())
    # print(image_lists.keys())
    # 读取已经训练好的Inception-v3模型。
    with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(
        graph_def, return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])

    # 定义新的神经网络输入
    bottleneck_input = tf.placeholder(tf.float32, [None, BOTTLENECK_TENSOR_SIZE], name='BottleneckInputPlaceholder')
    ground_truth_input = tf.placeholder(tf.float32, [None, n_classes], name='GroundTruthInput')

    # 定义一层全链接层
    with tf.name_scope('final_training_ops'):
        weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, n_classes], stddev=0.001))
        biases = tf.Variable(tf.zeros([n_classes]))
        logits = tf.matmul(bottleneck_input, weights) + biases
        final_tensor = tf.nn.softmax(logits)

    # 定义交叉熵损失函数。
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=ground_truth_input)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)

    # 计算正确率。
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(ground_truth_input, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 输出结果编码。
    saver = tf.train.Saver([weights], [biases])
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        saver.restore(sess, "data_set/model.ckpt")
        # 训练过程。
        #
        # for i in range(STEPS):
        #
        #     train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks(
        #         sess, n_classes, image_lists, BATCH, 'training', jpeg_data_tensor, bottleneck_tensor)
        #     sess.run(train_step,
        #              feed_dict={bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})
        #
        #     if i % 100 == 0 or i + 1 == STEPS:
        #         validation_bottlenecks, validation_ground_truth = get_random_cached_bottlenecks(
        #             sess, n_classes, image_lists, BATCH, 'validation', jpeg_data_tensor, bottleneck_tensor)
        #         validation_accuracy = sess.run(evaluation_step, feed_dict={
        #             bottleneck_input: validation_bottlenecks, ground_truth_input: validation_ground_truth})
        #         print('Step %d: Validation accuracy on random sampled %d examples = %.1f%%' %
        #               (i, BATCH, validation_accuracy * 100))
        #
        # # 在最后的测试数据上测试正确率。
        # test_bottlenecks, test_ground_truth = get_test_bottlenecks(
        #     sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor)
        # test_accuracy = sess.run(evaluation_step, feed_dict={
        #     bottleneck_input: test_bottlenecks, ground_truth_input: test_ground_truth})
        # print('Final test accuracy = %.1f%%' % (test_accuracy * 100))
        #
        # saver.save(sess, "data_set/model.ckpt")
        # show image
        # plt.imshow(INPUT_imgae)
        for i in [INPUT_target_1, INPUT_target_2, INPUT_target_3, INPUT_target_4, INPUT_target_5]:
            image_target = get_tensor(sess, i, jpeg_data_tensor, bottleneck_tensor)
            a = sess.run(tf.argmax(final_tensor, 1), feed_dict={bottleneck_input: image_target})
            font = ImageFont.truetype('/home/ck/Desktop/simHei.ttf', 24, encoding="UTF-8")
            image_1 = cv2.imread(i)
            image_2 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(image_2)
            draw = ImageDraw.Draw(pil_im)# 绘图句柄
            draw.text((0, 50), image_labels[int(a)], (255, 0, 0), font=font)
            cv2_text_im = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
            cv2.imshow(str(a), cv2_text_im)
            cv2.waitKey(0)

if __name__ == '__main__':
    main()