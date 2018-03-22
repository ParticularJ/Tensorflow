import os

# 兼容python2 与python3，xrange生成一个生成器
import six.moves import xrange
import tensorflow as tf


# Process images of this size
IMAGE_SIZE = 24

# Global constants describing the CIFAR-10 data set
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def read_cifar10(filename_queue):
# 如果你想要同时输入N个，那么调用这个函数N次
    """Returns:
        sigle example with the following fileds:
            height, width, depth, key:the filename & record number for this example.
            label, uint8image:a [height, width, depth]uint8 Tensor with the image data
    """

    class CIFAR10Record(object):
        pass
    result = CIFAR10Record()

    # input format
    label_bytes = 1 # 2 for CIFAR100
    result.height = 32
    result.width = 32
    rslult.depth = 3
    image_bytes = result.height * result.width * result.depth
    # Every record consists of a label followed by the image
    # fixed number of bytes for each
    record_bytes = label_bytes + image_bytes

    # Read a record, getting filenames from the filename_queue.
    # no header

    # A Reader that outputs fixed-length records from a file.
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)

    #Returns the next record (key, value) pair produced by a reader.
    result.key, value = reader.read(filename_queue)

    # Convert from a string to a vector of uint8 that is record_bytes long
    record_bytes = tf.decode_raw(value, tf.uint8)

    # The first bytes represent the label, which we convert from uint8->int32
    # Extracts a strided slice of a tensor(input_, begin, end, stride),提取（end-begin）/stride
    result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

    # The remaining bytes after the label represent the image, which
    # we reshape from [depth* height * width] to [depth, height, width]
    depth_major = tf.reshape(
        tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes]),
        [result.depth, result.height, result.width])

    # Convert from [depth, height, width] to [height, width, depth]
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    return result


def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
    """Construct a queued batch of images and labels.
    return:
        images:4-D Tensor of [batch_size, height, width, 3]
        labels:1D Tensor of [batch_size] size
    """

    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    if shuffle:
        # Creates batches by randomly shuffling tensors.
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            # 排列tensor_list的线程数
            num_threads=num_preprocess_threads,
            # the maximum number of elements in the queue.
            capacity=min_queue_examples + 3 * batch_size,
            # Minimum number elements in the queue after a dequeue,
            # used to ensure a level of mixing of elements.
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer
    tf.summary.image('images', images)

    return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(data_dir, batch_size):
    """Construct distorted input for CIFAR training using the Reader ops.
    """
    # xrange 生成一个生成器.
    # os.path.join 生成一个目录 data_dir/data_batch_%d.bin
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                for i in xrange(1, 6)]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_produce(filenames)

    with tf.name_scope('data_augmentataion'):
        # read example from files in the filename queue
        read_input = read_cifar10(filename_queue)
        reshaped_image = tf.cast(read_input.uint8image, tf.float32)

        height = IMAGE_SIZE
        width = IMAGE_SIZE

        # Image processing for training the network.

        # Randomly crop
        distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

        # randomly flip the image horizontally
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
        distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)

        # Subtract of the mean and divide by the variance of th pixels
        # 减去平均值并除以像素的方差。
        # Linearly scales image to have zero mean and unit norm.
        float_image = tf.image.per_image_standardization(distorted_image)

        # Set the shapes of tensors
        float_image.set_shape([height, width, 3])
        read_input.label.set_shape([1])

        # Ensure that the random shuffling has good mixing properties
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
        print('Filling queue with %d CIFAR images before starting to train. '
           'This will take a few minutes.' % min_queue_examples)

        # Generate a batch of images and labels by building up a queue of examples
        return _generate_image_and_label_batch(float_image, read_input.label,
                                                min_queue_examples, batch_size, shuffle=True)


def inputs(eval_data, data_dir, batch_size):
    """Construct input for CIFAR evalution using Reader ops."""

    if not eval_data:
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                    for i in xrange(1, 6)]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filenames = [os.path.join(data_dir, 'test_batch.bin')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file:' + f)

    with tf.name_scope('input'):

        # Output strings (e.g. filenames) to a queue for an input pipeline.
        filename_queue = tf.train.string_input_produce(filenames)

        read_input = read_cifar10(filename_queue)
        reshaped_image = tf.cast(read_input.uint8image, tf.float32)

        height = IMAGE_SIZE
        width = IMAGE_SIZE

        # Image precessing for evaluation
        # Crops and/or pads an image to a target width and height.
        resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, height, width)

        float_image = tf.image.per_image_standardization(resized_image)

        float_image.set_shape([height, width, 3])
        read_input.label.set_shape([1])

        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)

        return _generate_image_and_label_batch(float_image, read_input.label,
                                                min_queue_examples, batch_size, shuffle=True)
