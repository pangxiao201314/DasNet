from collections import namedtuple
import random
import numpy as np
import tensorflow as tf
import six

from tensorflow.python.training import moving_averages


HParams = namedtuple('HParams',
                     'batch_size, num_classes, min_lrn_rate, lrn_rate, '
                     'num_residual_units, use_bottleneck, weight_decay_rate, '
                     'relu_leakiness, optimizer, train_pa_flag, train_connect_flag, layer_keep')


class ResNet(object):

  def __init__(self, hps, mode, parameter_train_flag, connect_train_flag, connect_intial, input, output):

    self.hps = hps
    self._images = input
    self.labels = output
    self.mode = mode

    self.parameter_train_flag = parameter_train_flag
    self.connect_train_flag = connect_train_flag
    self.connect_inital = connect_intial
    self._extra_train_ops = []
    self.cardinality = 32
    self.cost_p = tf.placeholder(tf.float32, [1])
  # 构建模型图
  def build_graph(self):
    # 新建全局step
    self.global_step = tf.contrib.framework.get_or_create_global_step()
    # 构建ResNet网络模型
    self._build_model()
    # 构建优化训练操作
    #if self.mode == 'train':
      #self._build_train_op()
    # 合并所有总结
    #self.summaries = tf.summary.merge_all()


  # 构建模型
  def _build_model(self):
    with tf.variable_scope('init'):
      x = self._images
      """第一层卷积（3,7x7/2,64）"""
      x = self._conv('init_conv', x, 7, 3, 64, self._stride_arr(2))

    # 残差网络参数
    # bottleneck残差单元模块
    res_func = self._bottleneck_residual
    # 通道数量
    filters = [[64, 256], [128, 512], [256, 1024], [512, 2048]]

    residual_unit_count = 0
    # 第一组
    with tf.variable_scope('unit_1_0'):
      x = res_func(x=x, in_filter=64, first_two_out=128, out_filter=256, stride=self._stride_arr(1),
                   activate_before_residual=True, layer_keep=self.hps.layer_keep[residual_unit_count])
      residual_unit_count += 1

    for i in six.moves.range(1, self.hps.num_residual_units[0]):
      with tf.variable_scope('unit_1_%d' % i):
          x = res_func(x=x, in_filter=256, first_two_out=128, out_filter=256, stride=self._stride_arr(1),
                       activate_before_residual=False, layer_keep=self.hps.layer_keep[residual_unit_count])
          residual_unit_count += 1

    # 第二组
    with tf.variable_scope('unit_2_0'):
        x = res_func(x=x, in_filter=256, first_two_out=256, out_filter=512, stride=self._stride_arr(2),
                     activate_before_residual=False, layer_keep=self.hps.layer_keep[residual_unit_count])
        residual_unit_count += 1

    for i in six.moves.range(1, self.hps.num_residual_units[1]):
      with tf.variable_scope('unit_2_%d' % i):
          x = res_func(x=x, in_filter=512, first_two_out=256, out_filter=512, stride=self._stride_arr(1),
                       activate_before_residual=False, layer_keep=self.hps.layer_keep[residual_unit_count])
          residual_unit_count += 1
        
    # 第三组
    with tf.variable_scope('unit_3_0'):
        x = res_func(x=x, in_filter=512, first_two_out=512, out_filter=1024, stride=self._stride_arr(2),
                     activate_before_residual=False, layer_keep=self.hps.layer_keep[residual_unit_count])
        residual_unit_count += 1

    for i in six.moves.range(1, self.hps.num_residual_units[2]):
      with tf.variable_scope('unit_3_%d' % i):
          x = res_func(x=x, in_filter=1024, first_two_out=512, out_filter=1024, stride=self._stride_arr(1),
                       activate_before_residual=False, layer_keep=self.hps.layer_keep[residual_unit_count])
          residual_unit_count += 1

    # 第四组
    with tf.variable_scope('unit_4_0'):
        x = res_func(x=x, in_filter=1024, first_two_out=1024, out_filter=2048, stride=self._stride_arr(2),
                     activate_before_residual=False, layer_keep=self.hps.layer_keep[residual_unit_count])
        residual_unit_count += 1

    for i in six.moves.range(1, self.hps.num_residual_units[3]):
      with tf.variable_scope('unit_4_%d' % i):
        x = res_func(x=x, in_filter=2048, first_two_out=1024, out_filter=2048, stride=self._stride_arr(1),
                     activate_before_residual=False, layer_keep=self.hps.layer_keep[residual_unit_count])
        residual_unit_count += 1

    # 全局池化层
    with tf.variable_scope('unit_last'):
      x = self._batch_norm('final_bn', x)
      x = self._relu(x, self.hps.relu_leakiness)
      self.avg_pool = self._global_avg_pool(x)

    '''''''''
    # 全连接层 + Softmax
    with tf.variable_scope('logit'):
      logits = self._fully_connected(self.avg_pool, self.hps.num_classes)
      self.predictions = tf.nn.softmax(logits)

    # 构建损失函数
    with tf.variable_scope('costs'):
      # 交叉熵
      xent = tf.nn.softmax_cross_entropy_with_logits(
          logits=logits, labels=self.labels)
      # 加和
      self.cost = tf.reduce_mean(xent, name='xent')
      # L2正则，权重衰减
      if self.connect_train_flag:
          cost_p = self.cost_p
      else:
          cost_p = 1
      self.cost += cost_p * self._decay()

      # 添加cost总结，用于Tensorborad显示
      #tf.summary.scalar('cost', self.cost)
  '''''''''
  # 构建训练操作
  def _build_train_op(self):
    # 学习率/步长
    self.lrn_rate = tf.constant(self.hps.lrn_rate, tf.float32)
    tf.summary.scalar('learning_rate', self.lrn_rate)

    # 计算训练参数的梯度
    trainable_variables = tf.trainable_variables()
    grads = tf.gradients(self.cost, trainable_variables)

    # 设置优化方法
    if self.hps.optimizer == 'sgd':
      optimizer = tf.train.GradientDescentOptimizer(self.lrn_rate)
    elif self.hps.optimizer == 'mom':
      optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9)
    elif self.hps.optimizer == 'Adam':
      optimizer = tf.train.AdamOptimizer(self.lrn_rate)

    # 梯度优化操作
    range_update = []
    if self.connect_train_flag:
        for var_limit in tf.trainable_variables():
            range_update += tf.assign(var_limit, tf.clip_by_value(var_limit, 0.01, 1.0))

    apply_op = optimizer.apply_gradients(
                        zip(grads, trainable_variables),
                        global_step=self.global_step, 
                        name='train_step')

    # 合并BN更新操作
    train_ops = [range_update] + [apply_op] + self._extra_train_ops + [range_update]
    # 建立优化操作组
    self.train_op = tf.group(*train_ops)


  # 把步长值转换成tf.nn.conv2d需要的步长数组
  def _stride_arr(self, stride):    
    return [1, stride, stride, 1]

  # 残差单元模块
  def _residual(self, x, in_filter, out_filter, stride, activate_before_residual=False, layer_keep=True):
    # 是否前置激活(取残差直连之前进行BN和ReLU）
    if activate_before_residual:
      with tf.variable_scope('shared_activation'):
        # 先做BN和ReLU激活
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)
        # 获取残差直连
        orig_x = x
    else:
      with tf.variable_scope('residual_only_activation'):
        # 获取残差直连
        orig_x = x
        # 后做BN和ReLU激活
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)

    # 第1子层
    with tf.variable_scope('sub1'):
      # 3x3卷积，使用输入步长，通道数(in_filter -> out_filter)
      x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

    # 第2子层
    with tf.variable_scope('sub2'):
      # BN和ReLU激活
      x = self._batch_norm('bn2', x)
      x = self._relu(x, self.hps.relu_leakiness)
      # 3x3卷积，步长为1，通道数不变(out_filter)
      x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])
    
    # 合并残差层
    with tf.variable_scope('sub_add'):
      # 当通道数有变化时
      if in_filter != out_filter:
        # 均值池化，无补零
        orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
        # 通道补零(第4维前后对称补零)
        orig_x = tf.pad(orig_x, 
                        [[0, 0], 
                         [0, 0], 
                         [0, 0],
                         [(out_filter-in_filter)//2, (out_filter-in_filter)//2]
                        ])
      # 合并残差
      if layer_keep:
          connect_v = tf.get_variable('conncet_variable',
                                      [1],
                                      tf.float32,
                                      initializer=tf.constant_initializer(self.connect_inital),
                                      trainable=self.connect_train_flag)
      else:
          connect_v = tf.get_variable('conncet_variable',
                                      [0],
                                      tf.float32,
                                      initializer=tf.constant_initializer(self.connect_inital),
                                      trainable=False)

      x = tf.multiply(x, connect_v)
      x += orig_x

    tf.logging.debug('image after unit %s', x.get_shape())
    return x

  # bottleneck残差单元模块
  def _bottleneck_residual(self, x, in_filter, first_two_out, out_filter, stride,
                           activate_before_residual, layer_keep=True):
    # 是否前置激活(取残差直连之前进行BN和ReLU）
    if activate_before_residual:
      with tf.variable_scope('common_bn_relu'):
        # 先做BN和ReLU激活
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)
        x = tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=self._stride_arr(2), padding='SAME')
        # 获取残差直连
        orig_x = x
    else:
      with tf.variable_scope('residual_bn_relu'):
        # 获取残差直连
        orig_x = x
        # 后做BN和ReLU激活
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, self.hps.relu_leakiness)

    layers_split = list()
    for i in range(self.cardinality):
        if layer_keep[i]:
            with tf.variable_scope('_splitN_' + str(i)):
              splits = self.group_conv(x=x, in_filter=in_filter, first_two_out=first_two_out//self.cardinality,
                                       stride=stride, out_filter=out_filter, layer_keep_c=layer_keep[i])
              layers_split.append(splits)
    x = tf.add_n(layers_split)

      # 合并残差
    with tf.variable_scope('sub_add'):
        # 当通道数有变化时
        if in_filter != out_filter:
            # 1x1卷积，使用输入步长，通道数(in_filter -> out_filter)
            # orig_x = self._conv('project', orig_x, 1, in_filter, out_filter, stride)
            orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
            # 通道补零(第4维前后对称补零)
            font_pad = (out_filter - in_filter) // 2
            behind_pad = out_filter - in_filter - font_pad
            orig_x = tf.pad(orig_x,
                            [[0, 0],
                             [0, 0],
                             [0, 0],
                             [font_pad, behind_pad]
                             ])
        x += orig_x
    tf.logging.info('image after unit %s', x.get_shape())
    return x



  def group_conv(self, x, in_filter, first_two_out, stride, out_filter, layer_keep_c):

      # 第1子层
      with tf.variable_scope('sub1'):
          # 1x1卷积，使用输入步长，通道数(in_filter -> out_filter/4)
          x = self._conv('conv1', x, 1, in_filter, first_two_out, [1, 1, 1, 1])

      # 第2子层
      with tf.variable_scope('sub2'):
          # BN和ReLU激活
          x = self._batch_norm('bn2', x)
          x = self._relu(x, self.hps.relu_leakiness)
          # 3x3卷积，步长为1，通道数不变(out_filter/4)
          x = self._conv('conv2', x, 3, first_two_out, first_two_out, stride)
      # 第3子层
      with tf.variable_scope('sub3'):
          # BN和ReLU激活
          x = self._batch_norm('bn3', x)
          x = self._relu(x, self.hps.relu_leakiness)
          # 1x1卷积，步长为1，通道数不变(out_filter/4 -> out_filter)
          x = self._conv('conv3', x, 1, first_two_out, out_filter, [1, 1, 1, 1])

      # 合并残差层
      if layer_keep_c:
        connect_v = tf.get_variable('conncet_variable',
                                    [1],
                                    tf.float32,
                                    initializer=tf.constant_initializer(self.connect_inital),
                                    trainable=self.connect_train_flag)
      else:
        connect_v = tf.get_variable('conncet_variable',
                                    [0],
                                    tf.float32,
                                    initializer=tf.constant_initializer(self.connect_inital),
                                    trainable=False)

      return tf.multiply(x, connect_v)

  # Batch Normalization批归一化
  # ((x-mean)/var)*gamma+beta
  def _batch_norm(self, name, x):
    with tf.variable_scope(name):
      # 输入通道维数
      params_shape = [x.get_shape()[-1]]
      # offset
      beta = tf.get_variable('beta', 
                             params_shape, 
                             tf.float32,
                             initializer=tf.constant_initializer(0.0, tf.float32),
                             trainable= self.parameter_train_flag)
      # scale
      gamma = tf.get_variable('gamma', 
                              params_shape, 
                              tf.float32,
                              initializer=tf.constant_initializer(1.0, tf.float32),
                              trainable=self.parameter_train_flag)

      if self.mode == 'train':
        # 为每个通道计算均值、标准差
        # 新建或建立测试阶段使用的batch均值、标准差
        moving_mean = tf.get_variable('moving_mean', 
                                      params_shape, tf.float32,
                                      initializer=tf.constant_initializer(0.0, tf.float32),
                                      trainable=False)
        moving_variance = tf.get_variable('moving_variance', 
                                          params_shape, tf.float32,
                                          initializer=tf.constant_initializer(1.0, tf.float32),
                                          trainable=False)
        # 添加batch均值和标准差的更新操作(滑动平均)
        # moving_mean = moving_mean * decay + mean * (1 - decay)
        # moving_variance = moving_variance * decay + variance * (1 - decay)
        if self.connect_train_flag:
            mean = moving_mean
            variance = moving_variance
        else:
            mean, variance = tf.nn.moments(x, [0, 1, 2], name='moments')
            bn_v = tf.minimum(tf.Variable(0.99, dtype=tf.float64), 1-tf.divide(1, self.global_step+1))

            self._extra_train_ops.append(moving_averages.assign_moving_average(
                                                            moving_mean, mean, bn_v))
            self._extra_train_ops.append(moving_averages.assign_moving_average(
                                                            moving_variance, variance, bn_v))


      else:
        # 获取训练中积累的batch均值、标准差
        mean = tf.get_variable('moving_mean', 
                               params_shape, tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32),
                               trainable=False)
        variance = tf.get_variable('moving_variance', 
                                   params_shape, tf.float32,
                                   initializer=tf.constant_initializer(1.0, tf.float32),
                                   trainable=False)
        # 添加到直方图总结
        tf.summary.histogram(mean.op.name, mean)
        tf.summary.histogram(variance.op.name, variance)


      # BN层：((x-mean)/var)*gamma+beta
      y = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 0.001)
      y.set_shape(x.get_shape())
      return y


  # 权重衰减，L2正则loss
  def _decay(self):
    # 遍历所有可训练变量
    if self.connect_train_flag:
        cost_num = 0
        for var in tf.trainable_variables():
          var_trans = tf.convert_to_tensor(var)
          var_trans = tf.reshape(var_trans, [1,1])
          if cost_num == 0:
              costs = var_trans
          else:
            costs = tf.concat([costs, var_trans], 1)
          cost_num+=1
        #p_i = tf.nn.softmax(tf.log(costs))
        costs = tf.divide(costs,tf.reduce_sum(costs))
        cost_cv = tf.reduce_mean(-tf.reduce_sum(costs*tf.log(costs)))
        # 加和，并乘以衰减因子
        #return tf.multiply(self.hps.weight_decay_rate, tf.add_n(costs))
        return tf.reduce_mean(tf.multiply(1.0, cost_cv))
    else:
        costs = []
        for var in tf.trainable_variables():
          #只计算标有“DW”的变量
          if var.op.name.find(r'DW') > 0:
             costs.append(tf.nn.l2_loss(var))
        return tf.multiply(self.hps.weight_decay_rate, tf.add_n(costs))

  # 2D卷积
  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    with tf.variable_scope(name):
      n = filter_size * filter_size * out_filters
      # 获取或新建卷积核，正态随机初始化
      kernel = tf.get_variable(
              'DW', 
              [filter_size, filter_size, in_filters, out_filters],
              tf.float32, 
              initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)),
              trainable=self.parameter_train_flag)

      kernel_b = tf.get_variable(
              'DW_biase',
              [out_filters],
              tf.float32,
              initializer = tf.uniform_unit_scaling_initializer(factor=1.0),
              trainable=self.parameter_train_flag)
      # 计算卷积
      return tf.add(tf.nn.conv2d(x, kernel, strides, padding=('SAME' if strides[1]==1 else 'VALID')), kernel_b)

  # leaky ReLU激活函数，泄漏参数leakiness为0就是标准ReLU
  def _relu(self, x, leakiness=0.0):
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')
  
  # 全连接层，网络最后一层
  def _fully_connected(self, x, out_dim):
    # 输入转换成2D tensor，尺寸为[N,-1]
    num = x.get_shape()[1]
    x = tf.reshape(x, [self.hps.batch_size, -1])
    # 参数w，平均随机初始化，[-sqrt(3/dim), sqrt(3/dim)]*factor
    w = tf.get_variable('DW', [num , out_dim],
                        initializer=tf.uniform_unit_scaling_initializer(factor=1.0),
                        trainable = self.parameter_train_flag)
    # 参数b，0值初始化
    b = tf.get_variable('biases', [out_dim], initializer=tf.uniform_unit_scaling_initializer(factor=1.0),
                        trainable = self.parameter_train_flag)
    # 计算x*w+b
    return tf.nn.xw_plus_b(x, w, b)

  # 全局均值池化
  def _global_avg_pool(self, x):
    assert x.get_shape().ndims == 4
    # 在第2&3维度上计算均值，尺寸由WxH收缩为1x1
    return tf.reduce_mean(x, [1, 2])


def random_int_list(start, stop, length):
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for i in range(length):
        random_list.append(random.randint(start, stop))
    return random_list