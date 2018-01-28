import numpy as np
import tensorflow as tf
import cv2
import tqdm
from network import Network
import load
import matplotlib.pyplot as plt

import sys
sys.path.append('../data')
import write_tfrecord


IMAGE_SIZE = 128
LOCAL_SIZE = 64
HOLE_MIN = 24
HOLE_MAX = 48
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
PRETRAIN_EPOCH = 100
# PRETRAIN_EPOCH = 1

def train(backupFlag):
    x = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
    mask = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1])
    local_x = tf.placeholder(tf.float32, [BATCH_SIZE, LOCAL_SIZE, LOCAL_SIZE, 3])
    global_completion = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
    local_completion = tf.placeholder(tf.float32, [BATCH_SIZE, LOCAL_SIZE, LOCAL_SIZE, 3])
    is_training = tf.placeholder(tf.bool, [])

    model = Network(x, mask, local_x, global_completion, local_completion, is_training, batch_size=BATCH_SIZE)

    with tf.Session() as sess:
        # sess = tf.Session()
        global_step = tf.Variable(0, name='global_step', trainable=False)
        epoch = tf.Variable(0, name='epoch', trainable=False)

        opt = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        g_train_op = opt.minimize(model.g_loss, global_step=global_step, var_list=model.g_variables)
        d_train_op = opt.minimize(model.d_loss, global_step=global_step, var_list=model.d_variables)

        # init_op = tf.global_variables_initializer()
        # sess.run(init_op)

        if backupFlag and tf.train.get_checkpoint_state('./backup'):
            print('BACKING UP  LASTEST ===============================================')
            saver = tf.train.Saver()
            saver.restore(sess, './backup/latest')

        # loading with tfrecord
        print('READING TFRECORD for shape and image ===========================')
        tfrecord_filename = '../data/test.tfrecord'
        reader = tf.TFRecordReader()
        img, shape = write_tfrecord.read_tfrecord(reader,tfrecord_filename)

        # 1. need to seperate between test and train dataset
        # 2. change x_test,train to tf record type using hand gestures Example
        # 3. train
        # 4. check the shuffle batch examples

        step_num = int(1000 / BATCH_SIZE)

        # START COODRINATOR
        print('START COORDINATOR ===============================================')
        coord = tf.train.Coordinator()


        # SHUFFLE BATCH
        print('SHUFFLING BATCHES ===============================================')
        img_batch, shape_batch = tf.train.shuffle_batch(
                                        [img, shape],
                                        BATCH_SIZE,
                                        capacity=(10*BATCH_SIZE),
                                        min_after_dequeue=2*BATCH_SIZE,
                                        num_threads=4,
                                        enqueue_many = False)

        # NORMALIZE dataset
        # print('NORMALIZING DATASET =============================================')
        # img_batch = tf.cast(img_batch, tf.float32)
        # img_batch = img_batch / 127.5 - 1.0
        # half_twofivefive = tf.Variable(127.5, name = 'half_255',dtype = tf.float32)
        # neg_one = tf.Variable(-1, name='neg_one', dtype = tf.float32)
        # img_batch = tf.cast(img_batch, tf.float32)
        # img_batch = tf.div(img_batch, half_twofivefive,name='progress_norm')
        # img_batch = tf.subtract(img_batch,neg_one,name='norm_img_batch')


        init_local_op = tf.local_variables_initializer()
        init_op = tf.global_variables_initializer()
        sess.run([init_op,init_local_op])
        threads = tf.train.start_queue_runners(coord=coord)
        print('START TRAINING ==================================================')
        try:
            while not coord.should_stop():
        # ------------------------------------------------------------------------ #
                # sess.run([init_all_op])
                sess.run(tf.assign(epoch, tf.add(epoch, 1)))
                print('epoch: {}'.format(sess.run(epoch)))
                # sess.run([init_op,init_local_op])
                # np.random.shuffle(x_train)

                # Completion
                if sess.run(epoch) > 4*PRETRAIN_EPOCH:
                    print('breacking out of the loop')
                    break

                if sess.run(epoch) <= PRETRAIN_EPOCH:
                    g_loss_value = 0
                    print('COMPLETION NETWORK TRAINING PHASE')

                    for i in tqdm.tqdm(range(step_num)):

                        points_batch, mask_batch = get_points()
                        img_batch_seqs, shape_batch_seqs = sess.run([img_batch, shape_batch])
                        img_batch_seqs.reshape([BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,3])
                        # normalize image intensity for image batch_size
                        img_batch_seqs = normalize_image(img_batch_seqs, BATCH_SIZE)


                        shape_batch_seqs.reshape([4,3])


                        _, g_loss = sess.run(
                            [g_train_op, model.g_loss],
                            feed_dict={x: img_batch_seqs, mask: mask_batch, is_training: True})
                        g_loss_value += g_loss
                    print('Completion loss: {}'.format(g_loss_value))

                    # np.random.shuffle(x_test)
                    # x_batch = x_test[:BATCH_SIZE]
                    # completion = sess.run(model.completion, feed_dict={x: x_batch, mask: mask_batch, is_training: False})
                    img_batch_seqs = sess.run(img_batch)
                    # normalize image intensity for image batch_size
                    img_batch_seqs = normalize_image(img_batch_seqs, BATCH_SIZE)
                    completion = sess.run(
                        model.completion,
                        feed_dict = { x: img_batch_seqs,
                                    mask: mask_batch,
                                    is_training: False})
                    sample = np.array((completion[0] + 1) * 127.5, dtype=np.uint8)
                    cv2.imwrite('./output/{}.jpg'.format("{0:06d}".format(sess.run(epoch))), cv2.cvtColor(sample, cv2.COLOR_RGB2BGR))


                    saver = tf.train.Saver()
                    saver.save(sess, './backup/latest', write_meta_graph=False)
                    if sess.run(epoch) == PRETRAIN_EPOCH:
                        saver.save(sess, './backup/pretrained', write_meta_graph=False)


                            # Discrimitation
                else:
                    g_loss_value = 0
                    d_loss_value = 0
                    print('Discriminator NETWORK TRAINING PHASE')

                    for i in tqdm.tqdm(range(step_num)):

                        points_batch, mask_batch = get_points()
                        # print('\nmask batch shape: {}\npoints_batch shape: {}'.format(mask_batch.shape,points_batch.shape))
                        # _, g_loss, completion = sess.run([g_train_op, model.g_loss, model.completion], feed_dict={x: x_batch, mask: mask_batch, is_training: True})
                        # img_batch_seqs = sess.run(img_batch)
                        img_batch_seqs,shape_batch_seqs = sess.run([img_batch,shape_batch])
                        img_batch_seqs.reshape([BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,3])
                        shape_batch_seqs.reshape([BATCH_SIZE,3])
                        # normalize image intensity for image batch_size
                        img_batch_seqs = normalize_image(img_batch_seqs, BATCH_SIZE)

                        _, g_loss, completion = sess.run(
                            [g_train_op, model.g_loss, model.completion],
                            feed_dict={x: img_batch_seqs, mask: mask_batch, is_training: True})
                        g_loss_value += g_loss

                        local_x_batch = []
                        local_completion_batch = []


                        for i in range(BATCH_SIZE):
                            # ????? #
                            x1, y1, x2, y2 = points_batch[i]
                            local_x_batch.append(img_batch_seqs[i,y1:y2, x1:x2, :])
                            local_completion_batch.append(completion[i,y1:y2, x1:x2, :])

                        local_x_batch = np.array(local_x_batch)
                        local_completion_batch = np.array(local_completion_batch)


                        img_batch_seqs,shape_batch_seqs = sess.run([img_batch,shape_batch])
                        img_batch_seqs.reshape([BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,3])
                        shape_batch_seqs.reshape([BATCH_SIZE,3])
                        # normalize image intensity for image batch_size
                        img_batch_seqs = normalize_image(img_batch_seqs, BATCH_SIZE)
                        _, d_loss = sess.run(
                            [d_train_op, model.d_loss],
                            feed_dict={x:  img_batch_seqs,
                                    mask: mask_batch,
                                    local_x: local_x_batch,
                                    global_completion: completion,
                                    local_completion: local_completion_batch,
                                    is_training: True})
                        d_loss_value += d_loss
                    print('Completion loss: {}'.format(g_loss_value))
                    print('Discriminator loss: {}'.format(d_loss_value))

                            # np.random.shuffle(x_test)
                            # x_batch = x_test[:BATCH_SIZE]

                    img_batch_seqs,shape_batch_seqs = sess.run([img_batch,shape_batch])
                    img_batch_seqs.reshape([BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,3])
                    # normalize image intensity for image batch_size
                    img_batch_seqs = normalize_image(img_batch_seqs, BATCH_SIZE)
                    shape_batch_seqs.reshape([BATCH_SIZE,3])
                    completion = sess.run(
                    model.completion,
                    feed_dict={x: img_batch_seqs, mask: mask_batch, is_training: False})
                    # completion = sess.run(model.completion, feed_dict={x: x_batch, mask: mask_batch, is_training: False})
                    sample = np.array((completion[0] + 1) * 127.5, dtype=np.uint8)
                    cv2.imwrite('./output/{}.jpg'.format("{0:06d}".format(sess.run(epoch))), cv2.cvtColor(sample, cv2.COLOR_RGB2BGR))

                    saver = tf.train.Saver()
                    saver.save(sess, './backup/latest', write_meta_graph=False)

    # ------------------------------------------------------------------------ #

        except tf.errors.OutOfRangeError:
            print('ERROR: Done Training -- epoch limit reached')
        finally:
            print('FINAL: STOP COORDINATE and JOIN THREAD')
            coord.request_stop()
            coord.join(threads)
        # while True:


def get_points():
    points = []
    mask = []
    for i in range(BATCH_SIZE):
        # IMAGE SIZE: 128
        # LOCAL SIZE: 64
        # select a random point inside the image
        x1, y1 = np.random.randint(0, IMAGE_SIZE - LOCAL_SIZE + 1, 2)
        # select the other side of the points
        x2, y2 = np.array([x1, y1]) + LOCAL_SIZE
        # make a points array consisting of two points making a squar mask
        points.append([x1, y1, x2, y2])

        # HOLE_MAX: 48
        # HOLE_MIN: 24
        # select a random point inside the HOLE MIN MAX
        w, h = np.random.randint(HOLE_MIN, HOLE_MAX + 1, 2)
        p1 = x1 + np.random.randint(0, LOCAL_SIZE - w)
        q1 = y1 + np.random.randint(0, LOCAL_SIZE - h)
        p2 = p1 + w
        q2 = q1 + h

        # set the erased section to one
        m = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.uint8)
        m[q1:q2 + 1, p1:p2 + 1] = 1 # select the secdtion to be erased
        mask.append(m)


    return np.array(points), np.array(mask)

def normalize_image(image_batch, batch_size):
    for i in range(batch_size):
        norm_image = (image_batch[i] / 127.5) - 1
        norm_image_batch.append(norm_image)
if __name__ == '__main__':
    train(True)
