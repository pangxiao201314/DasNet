"""ResNet Train/Eval module.
"""
import time
import six
import os
import sys
import math
import numpy as np
import resnet_model
import threading
import tensorflow as tf
from revise_parameters import revise_conncet_VARIABLE
from test import c_t_b
from MobileNet import load_imagenet_meta, get_set_size, read_batch, read_validation_batch
from data_list import car_list, cat_list, dog_list
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# FLAGS参数设置
FLAGS = tf.app.flags.FLAGS
project_path = 'C:/Users/Deep Learning/PycharmProjects/ResNeXt_ImageNet'

# 模式：训练、测试
tf.app.flags.DEFINE_string('mode',
                           'train',
                           'train or eval.')
# 训练过程数据的存放路劲
tf.app.flags.DEFINE_string('train_dir',
                           project_path + '/train_temp',
                           'Directory to keep training outputs.')

tf.app.flags.DEFINE_string('eval_dir',
                           project_path + '/test_temp',
                           'Directory to keep training outputs.')
# 测试数据的Batch数量
tf.app.flags.DEFINE_integer('eval_batch_count',
                            1000,
                            'Number of batches to eval.')
# 一次性测试
tf.app.flags.DEFINE_bool('eval_once',
                         False,
                         'Whether evaluate the model only once.')
# 模型存储路劲
tf.app.flags.DEFINE_string('log_root',
                           project_path + '/train_connect_l50',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')
# GPU设备数量（0代表CPU）
tf.app.flags.DEFINE_integer('num_gpus',
                            1,
                            'Number of gpus used for training. (0 or 1)')


def train(hps):
    PATH = 'F:/ILSVRC/Data/CLS-LOC/'
    wnid_labels, _ = load_imagenet_meta(os.path.join(PATH, 'meta.mat'))
    train_set_size = get_set_size(os.path.join(PATH, 'train'))
    num_batches = int(float(train_set_size) / hps.batch_size)

    x = tf.placeholder(tf.float32, [None, 224, 224, 3])
    y = tf.placeholder(tf.float32, [None, 1000])

    with tf.device('/cpu:0'):
        q = tf.FIFOQueue(hps.batch_size * 3, [tf.float32, tf.float32], shapes=[[224, 224, 3], [1000]])
        enqueue_op = q.enqueue_many([x, y])
        x_b, y_b = q.dequeue_many(hps.batch_size)

    # 构建残差网络模型
    if hps.train_connect_flag:
        print('train connect parameters')
        model = resnet_model.ResNet(hps, FLAGS.mode, parameter_train_flag=False, connect_train_flag=True,
                                    connect_intial=0.1, input=x_b, output=y_b)
    else:
        print('train weights and biases parameters')
        model = resnet_model.ResNet(hps, FLAGS.mode, parameter_train_flag=True, connect_train_flag=False,
                                    connect_intial=1, input=x_b, output=y_b)
    model.build_graph()
    coord_read = tf.train.Coordinator()

    # 计算预测准确率
    truth = tf.argmax(model.labels, axis=1)
    predictions = tf.argmax(model.predictions, axis=1)
    precision = tf.reduce_mean(tf.to_float(tf.equal(predictions, truth)))

    # 建立总结存储器，每100步存储一次
    summary_hook = tf.train.SummarySaverHook(
        save_steps=1000000000,
        output_dir=FLAGS.train_dir,
        summary_op=tf.summary.merge(
            [model.summaries,
             tf.summary.scalar('Precision', precision)]))
    # 建立日志打印器，每100步打印一次
    logging_hook = tf.train.LoggingTensorHook(
        tensors={'step': model.global_step,
                 'loss': model.cost,
                 'precision': precision},
        every_n_iter=100)

    # 学习率更新器，基于全局Step
    class _LearningRateSetterHook(tf.train.SessionRunHook):

        def begin(self):
            # 初始学习率
            self._lrn_rate = 0.1

        def before_run(self, run_context):
            return tf.train.SessionRunArgs(
                # 获取全局Step
                model.global_step,
                # 设置学习率
                feed_dict={model.lrn_rate: self._lrn_rate})

        def after_run(self, run_context, run_values):
            # 动态更新学习率
            train_step = run_values.results
            if train_step < 70000:
                if model.connect_train_flag:
                    self._lrn_rate = 0.001
                else:
                    self._lrn_rate = 0.1

            elif train_step < 140000:
                if model.connect_train_flag:
                    self._lrn_rate = 0.001
                else:
                    self._lrn_rate = 0.05

            elif train_step < 200000:
                if model.connect_train_flag:
                    self._lrn_rate = 0.001
                else:
                    self._lrn_rate = 0.02

            elif train_step < 300000:
                if model.connect_train_flag:
                    self._lrn_rate = 0.001
                else:
                    self._lrn_rate = 0.008

            else:
                if model.connect_train_flag:
                    self._lrn_rate = 0.005
                else:
                    #self._lrn_rate = 0.001 * math.exp(-(train_step - 370000) / 10000)
                    self._lrn_rate = 0.001

                    # 建立监控Session

    if hps.train_pa_flag:
        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=FLAGS.log_root,
                save_checkpoint_secs=None,
                save_checkpoint_steps=5000,
                hooks=[_LearningRateSetterHook()],
                chief_only_hooks=[summary_hook],
                # 禁用默认的SummarySaverHook，save_summaries_steps and secs设置为None
                save_summaries_steps=None,
                save_summaries_secs=None,
                config=tf.ConfigProto(allow_soft_placement=True)) as mon_sess:

            def enqueue_batches():
                while not coord_read.should_stop():
                    im, l = read_batch(hps.batch_size, os.path.join(PATH, 'train'), wnid_labels)
                    mon_sess.run(enqueue_op, feed_dict={x: im, y: l})

            num_threads = 3
            for i in range(num_threads):
                t = threading.Thread(target=enqueue_batches)
                t.setDaemon(True)
                t.start()

            step_move = 0

            while not mon_sess.should_stop():
                step_move += 1
                # 执行训练操作
                mon_sess.run(model.train_op)
                if step_move%100 == 0:
                  epoch = step_move / num_batches
                  print('Epoch at present is: %.4f ;' % epoch, end=' ')
                  print('the accuracy now is: %.4f ;' % (mon_sess.run(precision)), end=' ')
                  print('the loss now is: %.4f ;' % (mon_sess.run(model.cost)))

    else:

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=FLAGS.log_root,
                save_checkpoint_secs=None,
                save_checkpoint_steps=None,
                hooks=[_LearningRateSetterHook()],
                chief_only_hooks=None,
                # 禁用默认的SummarySaverHook，save_summaries_steps and secs设置为None
                save_summaries_steps=None,
                save_summaries_secs=None,
                config=tf.ConfigProto(allow_soft_placement=True)) as mon_sess:

            def enqueue_batches():
                while not coord_read.should_stop():
                    im, l = read_batch(hps.batch_size, os.path.join(PATH, 'train'), wnid_labels)
                    mon_sess.run(enqueue_op, feed_dict={x: im, y: l})

            num_threads = 3
            for i in range(num_threads):
                t = threading.Thread(target=enqueue_batches)
                t.setDaemon(True)
                t.start()

            step_move = 0
            stop_train_connect = False

            while not mon_sess.should_stop():
                step_move += 1
                dynamic_cost = (10-0.01)/(1000000-1)*(step_move-1)+0.01
                if step_move >= 1000000:
                    dynamic_cost = 10
                mon_sess.run(model.train_op, feed_dict={model.cost_p: [dynamic_cost]})
                if step_move % 500 == 0:
                    print('the accuracy now is: %.4f ;' % (mon_sess.run(precision, feed_dict={model.cost_p: [dynamic_cost]})), end=' ')
                    print('the loss now is: %.4f ;' % (mon_sess.run(model.cost, feed_dict={model.cost_p: [dynamic_cost]})))
                    if step_move % 1000 == 0:
                        block_num = 0
                        connect_matrix = np.zeros((16, 32))
                        for i in range(4):
                            for j in range(0, hps.num_residual_units[i]):
                                with tf.variable_scope('unit_' + str(i + 1) + '_' + str(j), reuse=True):
                                    for k in range(32):
                                        with tf.variable_scope('_splitN_' + str(k), reuse=True):
                                            connect_v_train = tf.get_variable('conncet_variable', [1])
                                            connect_matrix[block_num, k] = mon_sess.run(connect_v_train)
                                block_num += 1
                        np.savetxt('connect', connect_matrix)
                        print('connect parameters are satisfied and have been saved in connect.txt')


def evaluate(hps):
    PATH = 'F:/ILSVRC/Data/CLS-LOC/'

    x = tf.placeholder(tf.float32, [None, 224, 224, 3])
    y = tf.placeholder(tf.float32, [None, 1000])

    with tf.device('/cpu:0'):
        q = tf.FIFOQueue(hps.batch_size * 3, [tf.float32, tf.float32], shapes=[[224, 224, 3], [1000]])
        enqueue_op = q.enqueue_many([x, y])
        x_b, y_b = q.dequeue_many(hps.batch_size)

    # 构建残差网络模型
    if hps.train_connect_flag:
        model = resnet_model.ResNet(hps, 'eval', parameter_train_flag=False, connect_train_flag=True,
                                    connect_intial=0.1, input=x_b, output=y_b)
    else:
        model = resnet_model.ResNet(hps, 'eval', parameter_train_flag=True, connect_train_flag=False,
                                    connect_intial=1, input=x_b, output=y_b)
    model.build_graph()
    coord = tf.train.Coordinator()
    # 模型变量存储器
    saver = tf.train.Saver()
    # 总结文件 生成器
    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir)

    # 执行Session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    def enqueue_batches():
        while not coord.should_stop():
            val_im, val_cls = read_validation_batch(hps.batch_size, os.path.join(PATH, 'validation'),
                                                    os.path.join(PATH, 'ILSVRC2012_validation_ground_truth.txt'))
            sess.run(enqueue_op, feed_dict={x: val_im, y: val_cls})

    num_threads = 5
    for i in range(num_threads):
        t = threading.Thread(target=enqueue_batches)
        t.setDaemon(True)
        t.start()

    best_precision = 0.0

    ckpt_state = tf.train.get_checkpoint_state(FLAGS.log_root)
    tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
    saver.restore(sess, ckpt_state.model_checkpoint_path)

    while True:
        # 逐Batch执行测试
        total_prediction, correct_prediction = 0, 0
        for _ in six.moves.range(FLAGS.eval_batch_count):
            # 执行预测
            (loss, predictions, truth, train_step) = sess.run(
                [model.cost, model.predictions,
                 model.labels, model.global_step])
            # 计算预测结果
            truth = np.argmax(truth, axis=1)
            predictions = np.argmax(predictions, axis=1)
            print(np.sum(truth == predictions)/hps.batch_size)
            correct_prediction += np.sum(truth == predictions)
            total_prediction += predictions.shape[0]
        print('the accuracy of validation dataset is %.5f' % (1.0*correct_prediction/total_prediction))
        # 计算准确率
        precision = 1.0 * correct_prediction / total_prediction
        best_precision = max(precision, best_precision)

        # 添加准确率总结
        precision_summ = tf.Summary()
        precision_summ.value.add(
            tag='Precision', simple_value=precision)
        summary_writer.add_summary(precision_summ, train_step)

        # 添加最佳准确总结
        best_precision_summ = tf.Summary()
        best_precision_summ.value.add(
            tag='Best Precision', simple_value=best_precision)
        summary_writer.add_summary(best_precision_summ, train_step)

        # 添加测试总结
        # summary_writer.add_summary(summaries, train_step)

        # 打印日志
        tf.logging.info('loss: %.3f, precision: %.3f, best precision: %.3f' %
                        (loss, precision, best_precision))

        # 执行写文件
        summary_writer.flush()

        if FLAGS.eval_once:
            break

        time.sleep(60)


def main(_):
    # 装逼，毫无营养的一段代码
    if FLAGS.num_gpus == 0:
        dev = '/cpu:0'
    elif FLAGS.num_gpus == 1:
        dev = '/gpu:0'
    else:
        raise ValueError('Only support 0 or 1 gpu.')

    # 每次执行train param, train connect , reconstruction 之后执行eval模式进行评估
    if FLAGS.mode == 'train':
        batch_size = 6
    elif FLAGS.mode == 'eval':
        batch_size = 60

    num_classes = 1000
    # the total layers of resnet is that 3*num_residual+2;
    residual_num = [3, 4, 6, 3]
    num_residual = 0
    for num_temp in residual_num:
        num_residual += num_temp

    # 初始化网络中保留的层，最开始每层都要保留下来
    layer_keep_list = []
    for _ in range(num_residual):
        cardinality_keep_list = []
        for _ in range(32):
            cardinality_keep_list.append(True)
        layer_keep_list.append(cardinality_keep_list)

    continue_remove = True

    while(continue_remove):
        # for step 1: train_pa = True, train_connect = False; for step 2: train_pa = False, train_connect = True;
        ##### Step 1 begin : 只训练参数
        train_pa = False
        train_connect = True
        if train_connect:
            optimze = 'sgd'
        else:
            optimze = 'mom'
        hps = resnet_model.HParams(batch_size=batch_size, num_classes=num_classes,
                                   min_lrn_rate=0.00001e-5, lrn_rate=0.1,
                                   num_residual_units=residual_num, use_bottleneck=True,
                                   weight_decay_rate=0.0001, relu_leakiness=0.0,
                                   optimizer=optimze, train_pa_flag=train_pa,
                                   train_connect_flag=train_connect, layer_keep=layer_keep_list)
        # 执行训练
        with tf.device(dev):
           train(hps)
        ################################# Step 1 end

        '''''''''
        ##### Evaluation begin :
        # 执行测试
        with tf.device(dev):
            accuracy = evaluate(hps)
            temp_accuracy = np.loadtxt('precision.txt')
            temp_accuracy = np.append(temp_accuracy, accuracy)
            np.savetxt('precision.txt', temp_ac0curacy)
        
        
        ######## Step 2 begin : 只训练连接系数
        ####连接系数变为0.3
        revise_conncet_VARIABLE(ckpt_path=FLAGS.log_root, num_residual=residual_num, keep_layer=layer_keep_list,
                                read_only=False, revise_value=0.3)
        train_pa = False
        train_connect = True
        if train_connect:
            optimze = 'sgd'
        else:
            optimze = 'mom'
        hps = resnet_model.HParams(batch_size=batch_size, num_classes=num_classes,
                                   min_lrn_rate=0.00001e-3, lrn_rate=0.1,
                                   num_residual_units=residual_num, use_bottleneck=False,
                                   weight_decay_rate=0.0002, relu_leakiness=0.0,
                                   optimizer=optimze, train_pa_flag=train_pa,
                                   train_connect_flag=train_connect, layer_keep=layer_keep_list)
        # 执行训练
        with tf.device(dev):
            train(hps)

        layer_keep_list, remove_num = c_t_b(np.loadtxt('connect'), layer_keep_list)
        ####连接系数变为1
        revise_conncet_VARIABLE(ckpt_path=FLAGS.log_root, num_residual=residual_num, keep_layer=layer_keep_list,
                                read_only=False, revise_value=1.0)

        if remove_num == 0:
            continue_remove = False

        ############################### Step 2 end
        '''''''''


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
