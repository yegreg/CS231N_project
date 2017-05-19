import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def main():
    # Loading in and dividing data
    # reads in data
    images = np.load("images.npy")
    pv_out = np.load("pv_outputs.npy")

    # randomly divides into a training set and a validation set
    num_imgs = images.shape[0]
    indices = np.arange(num_imgs)
    np.random.shuffle(indices)
    X_train, X_val = images[indices[:int(0.8 * num_imgs)]], images[indices[int(0.8 * num_imgs):]]
    y_train, y_val = pv_out[indices[:int(0.8 * num_imgs)]], pv_out[indices[int(0.8 * num_imgs):]]

    # Build computational graph
    tf.reset_default_graph()  # Reset computational graph

    x_var = tf.placeholder(tf.float32, [None, 60, 80, 3]) # x variable
    y_var = tf.placeholder(tf.float32, [None]) # y variable
    is_training = tf.placeholder(tf.bool) # flag
    pred_y_var = cnn_73_model(x_var, y_var, is_training) # model in use
    mean_loss = tf.losses.mean_squared_error(y_var, pred_y_var) # loss in use

    # Define optimizer and optimize session parameter
    # define optimizer
    optimizer = tf.train.AdamOptimizer(1e-3)
    # batch normalization in tensorflow requires this extra dependency
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train_step = optimizer.minimize(mean_loss)
    pass

    # Train the model
    # initialize all variable
    num_epochs = 30
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    # initialize loss and rel_err history list
    train_loss_hist = []
    train_error_hist = []
    val_error_hist = []

    for i in range(num_epochs):
        print('Training')
        train_loss, train_error = run_model(sess, pred_y_var, mean_loss,
                                            x_var,y_var,is_training,
                                            X_train, y_train, 1, 64, 100, train_step,False)
        print('Validation')
        val_loss, val_error = run_model(sess, pred_y_var, mean_loss,
                                        x_var, y_var, is_training, X_val, y_val)

        train_loss_hist.append(train_loss)
        train_error_hist.append(train_error)
        val_error_hist.append(val_error)

    # Plotting final results
    # plot training loss history
    plt.plot(train_loss_hist)
    plt.grid(True)
    plt.xlabel('Epoch number')
    plt.ylabel('training loss')
    plt.savefig('training_loss_history.png', bbox_inches='tight')
    plt.show()

    # plot relative error history
    plt.plot(train_error_hist[7:], label='training relative error')
    plt.plot(val_error_hist[7:], label='validation relative error')
    plt.grid(True)
    plt.xlabel('Epoch number')
    plt.ylabel('relative error')
    plt.legend()
    plt.savefig('relative_error.png', bbox_inches='tight')
    plt.show()


def run_model(session, pred_y_var, loss_var,
              x_var,y_var,is_training, Xd, yd,
              epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False):
    # have tensorflow compute accuracy
    rel_err_var = tf.divide(tf.abs(tf.subtract(y_var, pred_y_var)), y_var)
    accuracy = tf.reduce_mean(rel_err_var)

    # shuffle indices
    train_indices = np.arange(Xd.shape[0])
    np.random.shuffle(train_indices)
    training_now = training is not None

    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [loss_var, rel_err_var, accuracy]
    if training_now:
        variables[-1] = training

    # counter
    iter_cnt = 0
    for e in range(epochs):
        # keep track of losses and accuracy
        errors = 0
        losses = []
        # make sure we iterate over the dataset once
        for i in range(int(Xd.shape[0] / batch_size)+1):
            # generate indices for the batch
            start_idx = (i * batch_size) % Xd.shape[0]
            idx = train_indices[start_idx:start_idx + batch_size]

            # create a feed dictionary for this batch
            feed_dict = {x_var: Xd[idx, :],
                         y_var: yd[idx],
                         is_training: training_now}
            # get batch size
            actual_batch_size = yd[i:i + batch_size].shape[0]

            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            loss, rel_err, _ = session.run(variables, feed_dict=feed_dict)

            # aggregate performance stats
            losses.append(loss * actual_batch_size)
            errors += np.sum(rel_err)

            # print every now and then
            if training_now and (iter_cnt % print_every) == 0:
                print("Iteration {0}: with minibatch training loss = {1:.3g} and relative error of {2:.2g}" \
                      .format(iter_cnt, loss, np.sum(rel_err) / actual_batch_size))
            iter_cnt += 1
        total_error = errors / Xd.shape[0]
        total_loss = np.sum(losses) / Xd.shape[0]

        print("Epoch {2}, Overall loss = {0:.3g} and relative error of {1:.3g}" \
              .format(total_loss, total_error, e + 1))
        if plot_losses:
            plt.plot(losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e + 1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            plt.show()
    return total_loss, total_error


def cnn_73_model(X, y, is_training):
    # CBP sandwich 1
    conv1 = tf.layers.conv2d(
        inputs=X,
        filters=32,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)
    bn1 = tf.layers.batch_normalization(inputs=conv1, axis=1)
    pool1 = tf.layers.max_pooling2d(inputs=bn1, pool_size=[2, 2], strides=2)

    # CBP sandwich 2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu)
    bn2 = tf.layers.batch_normalization(inputs=conv2, axis=1)
    pool2 = tf.layers.max_pooling2d(inputs=bn2, pool_size=[2, 2], strides=2)

    # Two fully connected nets
    pool2_flat = tf.reshape(pool2, [-1, 20 * 15 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=is_training)
    regression = tf.layers.dense(inputs=dropout, units=1)
    regression = tf.reshape(regression, [-1])
    return regression


if __name__ == '__main__':
    main()
