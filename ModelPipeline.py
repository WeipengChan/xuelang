# -*- coding: utf-8 -*-、
# author : Robin
# 2017年12月07日18:02:12  保存模型机制优化,每次训练完后自动清理非最佳model

import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
import time
import matplotlib.pyplot as plt
models_path = "./models/"
from glob import glob
import os
GPU_COUNT = 4




#此model为菜品识别模型配置对象
class ModelConfig:
    """
    ModelConfig is a utility class that stores important configuration option about our model
    """

    def __init__(self, model, name, input_img_dimensions, conv_layers_config, fc_output_dims, output_classes,
                 dropout_keep_pct):
        self.model = model
        self.name = name
        self.input_img_dimensions = input_img_dimensions

        # Determines the wxh dimension of filters, the starting depth (increases by x2 at every layer)
        # and how many convolutional layers the network has
        self.conv_filter_size = conv_layers_config[0]
        self.conv_depth_start = conv_layers_config[1]
        self.conv_layers_count = conv_layers_config[2]

        self.fc_output_dims = fc_output_dims
        self.output_classes = output_classes

        # Try with different values for drop out at convolutional and fully connected layers
        self.dropout_conv_keep_pct = dropout_keep_pct[0]
        self.dropout_fc_keep_pct = dropout_keep_pct[1]


class ModelExecutor:
    """
    ModelExecutor is responsible for executing the supplied model
    """

    def __init__(self, model_config, learning_rate=0.001):
        self.model_config = model_config
        self.learning_rate = learning_rate

        self.init_graph()

    def init_graph(self):
        self.graph = tf.Graph()

        origin_model_config_name = self.model_config.name

        a = origin_model_config_name.split(".")
        if len(a) >= 2:
            origin_model_config_name = a[0][:-2] #如果是带了准确率的model名,必须去掉后面两位字符,才能符合刚训练时的model config name
            #print  origin_model_config_name

        with self.graph.as_default() as g:
            with g.name_scope(origin_model_config_name) as scope:
                # Create Model operations
                self.create_model_operations()

                # Create a saver to persist the results of execution
                self.saver = tf.train.Saver()

    def create_placeholders(self):
        """
        Defining our placeholder variables:
            - x, y
            - one_hot_y
            - dropout placeholders
        """

        # e.g. 32 * 32 * 3
        input_dims = self.model_config.input_img_dimensions
        self.x = tf.placeholder(tf.float32, (None, input_dims[0], input_dims[1], input_dims[2]),
                                name="{0}_x".format(self.model_config.name))
        self.y = tf.placeholder(tf.int32, (None), name="{0}_y".format(self.model_config.name))
        self.one_hot_y = tf.one_hot(self.y, self.model_config.output_classes)

        self.dropout_placeholder_conv = tf.placeholder(tf.float32)
        self.dropout_placeholder_fc = tf.placeholder(tf.float32)

    def create_model_operations(self):
        """
        Sets up all operations needed to execute run deep learning pipeline
        """

        # First step is to set our x, y, etc
        self.create_placeholders()

        cnn = self.model_config.model

        # Build the network -  TODO: pass the configuration in the future
        self.logits = cnn(self.x, self.model_config, self.dropout_placeholder_conv, self.dropout_placeholder_fc)
        # Obviously, using softmax as the activation function for final layer
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.one_hot_y, logits=self.logits)
        # Combined all the losses across batches
        self.loss_operation = tf.reduce_mean(self.cross_entropy)
        # What method do we use to reduce our loss?
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        # What do we really do in a training operation then? Answer: we attempt to reduce the loss using our chosen optimizer
        self.training_operation = self.optimizer.minimize(self.loss_operation)

        # Get the top prediction for model against labels and check whether they match
        self.correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.one_hot_y, 1))
        # Compute accuracy at batch level
        self.accuracy_operation = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        # compute what the prediction would be, when we don't have matching label
        self.prediction = tf.argmax(self.logits, 1)
        # Registering our top 5 predictions
        k = 5
        if self.model_config.output_classes < 5:
           k = self.model_config.output_classes
        self.top5_predictions = tf.nn.top_k(tf.nn.softmax(self.logits), k=k, sorted=True, name=None)

    def evaluate_model(self, X_data, Y_data, batch_size):
        """
        Evaluates the model's accuracy and loss for the supplied dataset.
        Naturally, Dropout is ignored in this case (i.e. we set dropout_keep_pct to 1.0)
        """

        num_examples = len(X_data)
        total_accuracy = 0.0
        total_loss = 0.0
        sess = tf.get_default_session()

        for offset in range(0, num_examples, batch_size):
            batch_x, batch_y = X_data[offset:offset + batch_size], Y_data[offset:offset + batch_size]

            # Compute both accuracy and loss for this batch
            accuracy = sess.run(self.accuracy_operation,
                                feed_dict={
                                    self.dropout_placeholder_conv: 1.0,
                                    self.dropout_placeholder_fc: 1.0,
                                    self.x: batch_x,
                                    self.y: batch_y
                                })
            loss = sess.run(self.loss_operation, feed_dict={
                self.dropout_placeholder_conv: 1.0,
                self.dropout_placeholder_fc: 1.0,
                self.x: batch_x,
                self.y: batch_y
            })

            # Weighting accuracy by the total number of elements in batch
            total_accuracy += (accuracy * len(batch_x))
            total_loss += (loss * len(batch_x))

            # To produce a true mean accuracy over whole dataset
        return (total_accuracy / num_examples, total_loss / num_examples)

    def train_model(self, X_train_features, X_train_labels, X_valid_features, y_valid_labels, batch_size=512,
                    epochs=100, PRINT_FREQ=10 , use_checkpoint=None):
        """
        Trains the model for the specified number of epochs supplied when creating the executor
        """

        # Create our array of metrics
        training_metrics = np.zeros((epochs, 3))
        validation_metrics = np.zeros((epochs, 3))

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        config.gpu_options.allow_growth = True

        with tf.Session(graph=self.graph,
                        config=config) as sess:
            if use_checkpoint != None:#如果要继续训练,自己输入上一次modelname
                model_file_name = "{0}{1}.chkpt".format(models_path, self.model_config.name)
                self.saver.restore(sess, model_file_name)

            sess.run(tf.global_variables_initializer())
            num_examples = len(X_train_features)

            print("Training {0} [epochs={1}, batch_size={2}]...\n".format(self.model_config.name, epochs, batch_size))

            self.best_acc = 0.0
            for i in range(0,epochs):
                #for d in range(GPU_COUNT):#即便我这样写也并没有并行运行
                    #with tf.device("/gpu:{0}".format(d)):
                        if i + 0 < epochs:
                            epoch_duration = self.epoch_extract(PRINT_FREQ, X_train_features, X_train_labels,
                                                          X_valid_features, batch_size, i, num_examples, sess,
                                                          training_metrics, validation_metrics, y_valid_labels)


            self.clean_models()

        return (training_metrics, validation_metrics, epoch_duration)

    def clean_models(self):
        print "cleaning models..."
        pattern = "{0}{1}_*.*".format(models_path, self.model_config.name)
        fileList = glob(pattern)
        best_acc = 0
        for infile in fileList:#先找出最大的准确率
            a = infile.split(".")
            if len(a) == 5:
                if a[1].split("_")[-1] == "1":
                    acc = "1.0"
                else:
                    acc = "0.{0}".format(a[2])

                acc = float(acc)
                if acc > best_acc:
                    best_acc = acc

        if best_acc > 0:  # 保留最好model,其它remove
            for infile in fileList:
                a = infile.split(".")
                if len(a) == 5:
                    if a[1].split("_")[-1] == "1":
                        acc = "1.0"
                    else:
                        acc = "0.{0}".format(a[2])
                    acc = float(acc)
                    if acc != best_acc:
                        try:
                            os.remove(infile)
                            print  "removing ", infile
                        except IOError:
                            print  "can not delete", infile

        print "models cleaned"

    def epoch_extract(self, PRINT_FREQ, X_train_features, X_train_labels, X_valid_features, batch_size, i, num_examples,
                    sess, training_metrics, validation_metrics, y_valid_labels):
        start = time.time()
        X_train, Y_train = shuffle(X_train_features, X_train_labels)
        for offset in range(0, num_examples, batch_size):
            end = offset + batch_size
            batch_x, batch_y = X_train[offset:end], Y_train[offset:end]
            sess.run(self.training_operation, feed_dict={
                self.x: batch_x,
                self.y: batch_y,
                self.dropout_placeholder_conv: self.model_config.dropout_conv_keep_pct,
                self.dropout_placeholder_fc: self.model_config.dropout_fc_keep_pct,

            })
        end_training_time = time.time()
        training_duration = end_training_time - start
        # computing training accuracy
        training_accuracy, training_loss = self.evaluate_model(X_train_features, X_train_labels, batch_size)
        # Computing validation accuracy
        validation_accuracy, validation_loss = self.evaluate_model(X_valid_features, y_valid_labels, batch_size)

        if validation_accuracy > self.best_acc:
            self.best_acc = validation_accuracy
            model_file_name = "{0}{1}_{2}.chkpt".format(models_path, self.model_config.name,self.best_acc)
            # Save the model
            self.saver.save(sess, model_file_name)
            print("Model {0} saved".format(model_file_name))

        end_epoch_time = time.time()
        validation_duration = end_epoch_time - end_training_time
        epoch_duration = end_epoch_time - start
        #if i == 0 or (i + 1) % PRINT_FREQ == 0:
        print("[{0}] total={1:.3f}s | train: time={2:.3f}s, loss={3:.4f}, acc={4:.4f} | val: time={5:.3f}s, loss={6:.4f}, acc={7:.4f}".format(
                    i + 1, epoch_duration, training_duration, training_loss, training_accuracy,
                    validation_duration, validation_loss, validation_accuracy))
        training_metrics[i] = [training_duration, training_loss, training_accuracy]
        validation_metrics[i] = [validation_duration, validation_loss, validation_accuracy]
        return epoch_duration

    def test_model(self, test_imgs, test_lbs, batch_size=512):
        """
        Evaluates the model with the test dataset and test labels
        Returns the tuple (test_accuracy, test_loss, duration)
        """

        with tf.Session(graph=self.graph) as sess:
            # Never forget to re-initialise the variables
            tf.global_variables_initializer()

            model_file_name = "{0}{1}.chkpt".format(models_path, self.model_config.name)
            self.saver.restore(sess, model_file_name)

            start = time.time()
            (test_accuracy, test_loss) = self.evaluate_model(test_imgs, test_lbs, batch_size)
            duration = time.time() - start
            print("[{0} - Test Set]\ttime={1:.3f}s, loss={2:.4f}, acc={3:.4f}".format(self.model_config.name, duration,
                                                                                      test_loss, test_accuracy))

        return (test_accuracy, test_loss, duration)

    #
    def predict(self, imgs, top_5=False):
        """
        Returns the predictions associated with a bunch of images
        """
        preds = None

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True #改默认为增长内存

        with tf.Session(graph=self.graph , config=config) as sess:
            # Never forget to re-initialise the variables
            tf.global_variables_initializer()

            model_file_name = "{0}{1}.chkpt".format(models_path, self.model_config.name)
            self.saver.restore(sess, model_file_name)

            if top_5:
                preds = sess.run(self.top5_predictions, feed_dict={
                    self.x: imgs,
                    self.dropout_placeholder_conv: 1.0,
                    self.dropout_placeholder_fc: 1.0
                })
            else:
                preds = sess.run(self.prediction, feed_dict={
                    self.x: imgs,
                    self.dropout_placeholder_conv: 1.0,
                    self.dropout_placeholder_fc: 1.0
                })

        return preds

    def show_conv_feature_maps(self, img, conv_layer_idx=0, activation_min=-1, activation_max=-1,
                               plt_num=1, fig_size=(15, 15), title_y_pos=1.0):
        """
        Shows the resulting feature maps at a given convolutional level for a SINGLE image
        """
        # s = tf.train.Saver()
        with tf.Session(graph=self.graph) as sess:
            # Never forget to re-initialise the variables
            tf.global_variables_initializer()
            # tf.reset_default_graph()

            model_file_name = "{0}{1}.chkpt".format(models_path, self.model_config.name)
            self.saver.restore(sess, model_file_name)

            # Run a prediction
            preds = sess.run(self.prediction, feed_dict={
                self.x: np.reshape(img, [1, img.shape[0], img.shape[1], img.shape[2]]),
                self.dropout_placeholder_conv: 1.0,
                self.dropout_placeholder_fc: 1.0
            })

            var_name = "{0}/conv_{1}_relu:0".format(self.model_config.name, conv_layer_idx)
            print("Fetching tensor: {0}".format(var_name))
            conv_layer = tf.get_default_graph().get_tensor_by_name(var_name)

            activation = sess.run(conv_layer, feed_dict={
                self.x: np.reshape(img, [1, img.shape[0], img.shape[1], img.shape[2]]),
                self.dropout_placeholder_conv: 1.0,
                self.dropout_placeholder_fc: 1.0
            })
            featuremaps = activation.shape[-1]
            # (1, 13, 13, 64)
            print("Shape of activation layer: {0}".format(activation.shape))

            # fix the number of columns
            cols = 8
            rows = featuremaps // cols
            fig, axes = plt.subplots(rows, cols, figsize=fig_size)
            k = 0
            for i in range(0, rows):
                for j in range(0, cols):
                    ax = axes[i, j]
                    featuremap = k

                    if activation_min != -1 & activation_max != -1:
                        ax.imshow(activation[0, :, :, featuremap], interpolation="nearest", vmin=activation_min,
                                  vmax=activation_max, cmap="gray")
                    elif activation_max != -1:
                        ax.imshow(activation[0, :, :, featuremap], interpolation="nearest", vmax=activation_max,
                                  cmap="gray")
                    elif activation_min != -1:
                        ax.imshow(activation[0, :, :, featuremap], interpolation="nearest", vmin=activation_min,
                                  cmap="gray")
                    else:
                        ax.imshow(activation[0, :, :, featuremap], interpolation="nearest", cmap="gray")

                    ax.axis("off")
                    k += 1

            fig.suptitle("Feature Maps at layer: {0}".format(conv_layer), fontsize=12, fontweight='bold', y=title_y_pos)
            fig.tight_layout()
            plt.show()

#if __name__ == '__main__':
