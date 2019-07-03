def cnn_model_fn(features, labels, mode):
    # tf.reset_default_graph()
    input_layer = tf.reshape(features["x"], [-1, 1, 28, 28])
    input_layer = tf.transpose(input_layer, perm=[0, 2, 3, 1])
    # conv1_1： 神经元图， feature_map, 输出图像
    conv1_1 = tf.layers.conv2d(input_layer,
                               28,  # output channel number
                               (3, 3),  # kernel size
                               padding='valid',
                               activation=tf.nn.relu,
                               name='conv1_1')
    # conv1_2： 神经元图， feature_map, 输出图像
    conv1_2 = tf.layers.conv2d(conv1_1,
                               28,  # output channel number
                               (3, 3),  # kernel size
                               padding='valid',
                               activation=tf.nn.relu,
                               name='conv1_2')

    # 14 * 14
    pooling1 = tf.layers.max_pooling2d(conv1_2,
                                       (2, 2),  # kernel size
                                       (2, 2),  # stride
                                       name='pool1')

    # conv2_1： 神经元图、 feature_map, 输出图像
    conv2_1 = tf.layers.conv2d(pooling1,
                               28,  # output channel number
                               (3, 3),  # kernel size
                               padding='valid',
                               activation=tf.nn.relu,
                               name='conv2_1')
    # conv2_2： 神经元图、 feature_map, 输出图像
    conv2_2 = tf.layers.conv2d(conv2_1,
                               28,  # output channel number
                               (3, 3),  # kernel size
                               padding='valid',
                               activation=tf.nn.relu,
                               name='conv2_2')

    # 7 * 7
    pooling2 = tf.layers.max_pooling2d(conv2_2,
                                       (2, 2),  # kernel size
                                       (2, 2),  # stride
                                       name='pool2')

    flatten = tf.layers.flatten(pooling2)
    dense = tf.layers.dense(inputs=flatten, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)
    # 全连接层,输出
    logits = tf.layers.dense(inputs=dropout, units=26)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        # sess = tf.InteractiveSession()
        # print('logits:',logits.eval(session = sess))
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # cross entropy
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    # y_ -> softmax
    # y -> one_hot
    # loss = ylogy_

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    # Add evaluation metrics (for EVAL mode)

    with tf.name_scope('train_op'):
        train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
        # 梯度下降的变种