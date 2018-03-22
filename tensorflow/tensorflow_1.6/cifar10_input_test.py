import os

import tensorflow as tf

import cifar10_input

# Base class for tests that need to test TensorFlow.
class CIFAR10InputTest(tf.test.TestCase):

    def _record(self, label, red, green, blue):
        image_size = 32 * 32
        # 生成bytes
        record = bytes(bytearray([label] + [red] * image_size +
                                 [green] * image_size + [blue] * image_size))
        #
        expected = [[[red, green, blue]] * 32] * 32
        return record, expected

    def testSimple(self):
        labels = [9, 3, 0]
        records = [self._record(labels[0], 0, 128, 255),
                   self._record(labels[1], 255, 0, 1),
                   self._record(labels[2], 254, 255, 0)]
        contents = b"".join([record for record, _ in records])
        expected = [expected for _, expected in records]
        filename = os.path.join(self.get_temp_dir(), "cifar")
        open(filename, "wb").write(contents)

        with self.test_session() as sess:
            # A queue implementation that dequeues elements in first-in first-out order.
            # capacity:99 The upper bound on the number of elements that may be stored in this queue.
            q = tf.FIFOQueue(99, [tf.string], shapes=())

            # Enqueues one element to this queue.
            q.enqueue([filename]).run()
            q.close().run()
            result = cifar10_input.read_cifar10(q)

            for i in range(3):
                key, label, uint8image = sess.run([
                    result.key, result.label, result.uint8image])
            #  check for an expected result
            # tf.compat.as_text() Returns the given argument as a unicode string.
            self.assertEqual("%s:%d" % (filename, i), tf.compat.as_text(key))
            self.assertEqual(labels[i], label)
            self.assertAllEqual(expected[i], uint8image)

        with self.assertRaises(tf.errors.OutOfRangeError):
            sess.run([result.key, result.uint8image])


if __name__ == "__main__":
    tf.test.main()
