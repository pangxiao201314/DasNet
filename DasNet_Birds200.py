import random
import numpy as np
import resnet_model
import tensorflow as tf
from MobileNet import preprocess_image
from test import compute_save_para

def read(train_or_test):
    f_train = open(TRAIN_DIR + train_or_test, "r")
    line = True
    img = []
    label = []
    while line:
        line = f_train.readline()
        line = line[:-1]
        if line != '':
            img.append(line.split(' ')[1])
            label.append(onehot(int(line.split(' ')[0])))
    return img, label

def read_img_batch(img_list, label_list, random, train_flag):
    batch_img = []
    batch_label = []
    for i in random:
        batch_img.append(read_img(img_list[i], train_flag=train_flag))
        batch_label.append(label_list[i])
    return batch_img, batch_label

def read_img(img_dir, train_flag):
    return preprocess_image(TRAIN_DIR + '/images/' + img_dir, train_flag=train_flag)

def onehot(index):
    onehot = np.zeros(200)
    onehot[index-1] = 1.0
    return onehot

def random_int_list(start, stop, length):
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for i in range(length):
        random_list.append(random.randint(start, stop))
    return random_list

def classify_net(input, train_flag):
    net = {}
    net['input'] = input
    w = tf.get_variable('classify_W', [2048, 200],
                        initializer=tf.uniform_unit_scaling_initializer(factor=1.0), trainable=train_flag)
    b = tf.get_variable('classify_biase', [200], initializer=tf.uniform_unit_scaling_initializer(factor=1.0), trainable=train_flag)
    net['fully_connected'] = tf.nn.xw_plus_b(net['input'], w, b)
    predictions = tf.nn.softmax(net['fully_connected'])
    net['out'] = predictions
    return net, [w, b]

def get_uninitialized_variables(sess):
   global_vars = tf.global_variables()
   is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
   not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
   #print([str(i.name) for i in not_initialized_vars])
   return not_initialized_vars

batch_size = 6

connect_train_flag = True
fine_tune_flag = False

if connect_train_flag:
    op = 'sgd'
else:
    op = 'mom'

project_path = 'C:/Users/Deep Learning/PycharmProjects/ResNeXt_ImageNet'
TRAIN_DIR = "E:/Birds_200"

residual_num = [3, 4, 6, 3]
num_residual = 0
for num_temp in residual_num:
    num_residual += num_temp

layer_keep_list = []
connect_gate_value = 0
connect = np.loadtxt(project_path+'/connect_Birds/connect_1')
compute_save_para(gate_value=connect_gate_value, connect_name=project_path+'/connect_Birds/connect_1')
for row in range(num_residual):
    cardinality_keep_list = []
    for col in range(32):
        if connect[row, col] >= connect_gate_value:
            cardinality_keep_list.append(True)
        else:
            cardinality_keep_list.append(False)
    layer_keep_list.append(cardinality_keep_list)


hps = resnet_model.HParams(batch_size=batch_size, num_classes=1000,
                           min_lrn_rate=0.00001e-5, lrn_rate=0.1,
                           num_residual_units=residual_num, use_bottleneck=True,
                           weight_decay_rate=0.0001, relu_leakiness=0.0,
                           optimizer=op, train_pa_flag=False,
                           train_connect_flag=connect_train_flag, layer_keep=layer_keep_list)

x = tf.placeholder(tf.float32, [None, 224, 224, 3])
y = tf.placeholder(tf.float32, [None, 1000])

model = resnet_model.ResNet(hps, 'eval', parameter_train_flag=False, connect_train_flag=connect_train_flag,
                            connect_intial=1, input=x, output=y)
model.build_graph()
saver = tf.train.Saver()

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
ckpt_state = tf.train.get_checkpoint_state(project_path+'/train_connect_l50')
tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
sess.run(tf.global_variables_initializer())
saver.restore(sess, ckpt_state.model_checkpoint_path)

classify_model, var_list = classify_net(model.avg_pool, train_flag=fine_tune_flag)
saver2 = tf.train.Saver(var_list=var_list)
ckpt_state = tf.train.get_checkpoint_state(project_path + '/connect_Birds/Birds_weights')
tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
saver.restore(sess, ckpt_state.model_checkpoint_path)

labels = tf.placeholder(tf.float32, [None, 200])
lr = tf.placeholder(tf.float32)
alpha = tf.placeholder(tf.float32)

xent = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=classify_model['fully_connected'], label_smoothing=0.0)
loss = tf.reduce_mean(xent)
if connect_train_flag:
    cost_num = 0
    for var in tf.trainable_variables():
        var_trans = tf.convert_to_tensor(var)
        var_trans = tf.reshape(var_trans, [1, 1])
        if cost_num == 0:
            costs = var_trans
        else:
            costs = tf.concat([costs, var_trans], 1)
        cost_num += 1
    costs = tf.divide(costs, tf.reduce_sum(costs))
    cost_cv = tf.reduce_mean(-tf.reduce_sum(costs * tf.log(costs)))
    loss += alpha*tf.reduce_mean(tf.multiply(1.0, cost_cv))

trainable_variables = tf.trainable_variables()
grads = tf.gradients(loss, trainable_variables)
optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(loss)
range_update = []
if connect_train_flag:
    for var_limit in tf.trainable_variables():
        range_update += tf.assign(var_limit, tf.clip_by_value(var_limit, 0.01, 1.0))
apply_op = optimizer.apply_gradients(
                        zip(grads, trainable_variables),
                        name='train_step')
train_ops = [range_update] + [apply_op] + [range_update]
train_op = tf.group(*train_ops)


img_train, train_label = read('/train.txt')
train_total = len(img_train)
img_test, test_label = read('/test.txt')
test_total = len(img_test)

#sess.run(tf.variables_initializer(get_uninitialized_variables(sess)))
step = 0
display= 500
test_dis = 2001
'''''''''
while True:
    step+=1

    if step <= 5000:
        learning = 0.05
    elif step <= 10000:
        learning = 0.01
    else:
        learning = 0.001

    ran = random_int_list(0, train_total-1, batch_size)
    img_b, label_b = read_img_batch(img_train, train_label, ran, train_flag=True)

    if step%display ==0:
        predictions = sess.run(classify_model['out'], feed_dict={x:img_b})
        loss_value = sess.run(loss, feed_dict={x:img_b, labels:label_b})
        truth = np.argmax(label_b, axis=1)
        predictions = np.argmax(predictions, axis=1)
        ac = np.sum(truth == predictions) / batch_size
        print('the accuracy now on train set is %.4f, ' % (ac), end=' ')
        print('and the loss now on train set is %.4f.' % (loss_value))
        loss_list_temp = np.loadtxt('loss_list')
        loss_list_temp = np.append(loss_list_temp, loss_value)
        np.savetxt('loss_list', loss_list_temp)

    elif step%test_dis == 0:
        ran = random_int_list(0, test_total - 1, batch_size)
        img_b, label_b = read_img_batch(img_test, test_label, ran, train_flag=False)
        predictions = sess.run(classify_model['out'], feed_dict={x: img_b})
        truth = np.argmax(label_b, axis=1)
        predictions = np.argmax(predictions, axis=1)
        ac = np.sum(truth == predictions) / batch_size
        print('the accuracy now on test set is %.4f ' % (ac))

    else:
        sess.run(optimizer, feed_dict={x:img_b, labels:label_b, lr:learning})
'''''''''''

while True:
    step += 1
    ran = random_int_list(0, train_total - 1, batch_size)
    img_b, label_b = read_img_batch(img_train, train_label, ran, train_flag=False)
    alpha_real = (10 - 0.01) / (1000000 - 1) * (step - 1) + 0.01
    if step >= 1000000:
        alpha_real = 10
    sess.run(train_op, feed_dict={x:img_b, labels:label_b, lr:0.005, alpha: [alpha_real]})
    if step % 500 == 0:
        predictions = sess.run(classify_model['out'], feed_dict={x: img_b})
        loss_value = sess.run(loss, feed_dict={x: img_b, labels: label_b})
        truth = np.argmax(label_b, axis=1)
        predictions = np.argmax(predictions, axis=1)
        ac = np.sum(truth == predictions) / batch_size
        print('the accuracy now on train set is %.4f, ' % (ac), end=' ')
        print('and the loss now on train set is %.4f.' % (loss_value))
        if step % 1000 == 0:
            block_num = 0
            connect_matrix = np.zeros((16, 32))
            for i in range(4):
                for j in range(0, hps.num_residual_units[i]):
                    with tf.variable_scope('unit_' + str(i + 1) + '_' + str(j), reuse=True):
                        for k in range(32):
                            with tf.variable_scope('_splitN_' + str(k), reuse=True):
                                connect_v_train = tf.get_variable('conncet_variable', [1])
                                connect_matrix[block_num, k] = sess.run(connect_v_train)
                    block_num += 1
            np.savetxt('/connect_Birds/connect', connect_matrix)
            print('connect parameters are satisfied and have been saved in connect.txt')



