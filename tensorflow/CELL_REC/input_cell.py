import tensorflow as tf
import os

FLAGS = tf.app.flags.FLAGS

IMAGE_SIZE = 24
tf.app.flags.DEFINE_string('train_dir', "/home/ck/cell_DL/CELL_images/train_data_trans", 'The dir of train')
# tf.app.flags.DEFINE_integer('batch_size', 2, "the number of batch_size")
SPECIES = ['HSIL', 'LSIL', 'NILM']


def inputs():

	filename = [os.path.join(FLAGS.train_dir, f) for f in tf.gfile.ListDirectory(FLAGS.train_dir)]
	label = []
	for f in filename:
		name = f.split('/')[-1].split('_')[0]
		if name == "HSIL":
			label.append("0")
		if name == "LSIL":
			label.append("1")
		if name == "NILM":
			label.append("2")
	filenames = tf.constant(filename, dtype=tf.string)
	labels = tf.constant(label)
	dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
	dataset = dataset.map(_parse_funtion)

	iterator = dataset.make_one_shot_iterator()
	one_element = iterator.get_next()
	with tf.Session() as sess:
		try:
			while True:
				sess.run(one_element)
				# print(sess.run(dataset))
		except tf.errors.OutOfRangeError:
			print("end!")
	return dataset

def _parse_funtion(filename, label):
	height = IMAGE_SIZE
	width = IMAGE_SIZE

	image_string = tf.read_file(filename)
	image_decoded = tf.image.decode_jpeg(image_string, channels=3)
	resize_img = tf.image.resize_images(image_decoded, [height, width])
	# resize_img = tf.image.resize_image_with_crop_or_pad(image_decoded, height, width)
	float_image = tf.cast(resize_img, tf.float32)
	# float_image = tf.image.per_image_standardization(resize_img)
	return float_image, label


def main(argv=None):
	dataset = inputs()
	# print(dataset)



if __name__ == '__main__':
	tf.app.run()