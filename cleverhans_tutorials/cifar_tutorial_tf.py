"""
This tutorial shows how to generate some simple adversarial examples
and train a model using adversarial training using nothing but pure
TensorFlow.
It is very similar to mnist_tutorial_keras_tf.py, which does the same
thing but with a dependence on keras.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from keras.utils import np_utils
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
import logging

from cleverhans.utils_mnist import data_mnist
from keras.datasets import cifar10

from cleverhans.utils_tf import model_train, model_eval,model_loss,initialize_uninitialized_global_variables
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import SaliencyMapMethod
from cleverhans.attacks import ElasticNetMethod
from cleverhans_tutorials.tutorial_models import make_basic_cnn,make_resnet,make_strong_cnn
from cleverhans.utils import AccuracyReport, set_log_level

import os

FLAGS = flags.FLAGS


baseDir = os.path.dirname(os.path.abspath('__file__')) + '/'
classesName = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', '\
ship', 'truck']

def cifar_tutorial(train_start=0, train_end=49000, test_start=0,
                   test_end=10000, nb_epochs=6, batch_size=128,
                   learning_rate=0.001,
                   clean_train=True,
                   testing=False,
                   backprop_through_attack=False,
                   nb_filters=64, num_threads=None):
    """
    MNIST cleverhans tutorial
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param clean_train: perform normal training on clean examples only
                        before performing adversarial training.
    :param testing: if true, complete an AccuracyReport for unit tests
                    to verify that performance is adequate
    :param backprop_through_attack: If True, backprop through adversarial
                                    example construction process during
                                    adversarial training.
    :param clean_train: if true, train on clean examples
    :return: an AccuracyReport object
    """

    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    # Set logging level to see debug information
    set_log_level(logging.DEBUG)

    # Create TF session
    if num_threads:
        config_args = dict(intra_op_parallelism_threads=1)
    else:
        config_args = {}
    sess = tf.Session(config=tf.ConfigProto(**config_args))


    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    Y_train = np_utils.to_categorical(Y_train, 10)
    Y_test =  np_utils.to_categorical(Y_test, 10)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    label_smooth = .1
    
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)
    
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))

    y = tf.placeholder(tf.float32, shape=([None,10]))


    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }
    fgsm_params = {'eps': FLAGS.fgsm_eps,
                   'clip_min': 0.,
                   'clip_max': 1.}
    rng = np.random.RandomState([2017, 8, 30])
    prune_factor = FLAGS.prune_factor
    conv_prune_factor = 5
    if clean_train:
        prune_percent = {'conv1_w':5,'conv2_w':5,'conv3_w':5,'conv4_w':5,'fc1_w':10,'fc2_w':10,'fc3_w':10}
        #model = make_resnet(x,10,[None,32,32,3],reuse = True,prune_percent = prune_percent)
        model = make_strong_cnn(nb_filters=nb_filters,prune_percent=prune_percent)
        initialize_uninitialized_global_variables(sess)
        preds = model.get_probs(x)
        saver = tf.train.Saver()
        eval_par = {'batch_size': batch_size}        
        def evaluate():
            # Evaluate the accuracy of the MNIST model on legitimate test
            # examples
            eval_params = {'batch_size': batch_size}
            acc = model_eval(
                sess, x, y, preds, X_test, Y_test, args=eval_params)
            report.clean_train_clean_eval = acc
            assert X_test.shape[0] == test_end - test_start, X_test.shape
            print('Test accuracy on legitimate examples: %0.4f' % acc)

        ckpt_name = './cifar_model.ckpt'
        
        if not FLAGS.resume:
            model_train(sess, x, y, preds, X_train, Y_train, evaluate=evaluate,
                        args=train_params, rng=rng)
            saver.save(sess,ckpt_name)
        if FLAGS.resume:
            saver = tf.train.import_meta_graph(ckpt_name+'.meta')
            print ("loading pretrain model")
            saver.restore(sess,ckpt_name)
            acc = model_eval(sess, x, y, preds, X_test, Y_test, args=eval_par)
            print('Test accuracy on pretrained model: %0.4f\n' % acc)
        if not FLAGS.resume:
            import sys
            sys.exit()
        
        # Initialize the Fast Gradient Sign Method (FGSM) attack object and
        # graph
        
        fgsm = FastGradientMethod(model, sess=sess)
        model.test_mode()
        #sess.run(tf.global_variables_initializer())
        initialize_uninitialized_global_variables(sess)
        eval_par = {'batch_size': batch_size}
        adv_x = fgsm.generate(x, **fgsm_params)
        
        preds_adv = model.get_probs(adv_x)
        
        acc = model_eval(sess, x, y, preds_adv, X_test, Y_test, args=eval_par)
        print('Initially, Test accuracy on adversarial examples: %0.4f\n' % acc)
        
        # Calculate training error
        if testing:
            eval_par = {'batch_size': batch_size}
            acc = model_eval(sess, x, y, preds_adv, X_train,
                             Y_train, args=eval_par)
            report.train_clean_train_adv_eval = acc


        if not FLAGS.load_pruned_model:
            print ("start iterative pruning")
            for i in range(FLAGS.prune_iterations):
                print ("iterative %d"  % (i))
                dict_nzidx = model.apply_prune(sess)
                trainer = tf.train.AdamOptimizer(learning_rate)
                preds = model.get_probs(x)
                loss = model_loss(y,preds)            
                grads = trainer.compute_gradients(loss)            
                grads = model.apply_prune_on_grads(grads,dict_nzidx)
                prune_args = {'trainer':trainer,'grads':grads}
                train_params = {
                    'nb_epochs':FLAGS.retrain_epoch,
                    'batch_size': batch_size,
                    'learning_rate': FLAGS.retrain_lr
                    }
                model_train(sess, x, y, preds, X_train, Y_train, evaluate=evaluate,
                            args=train_params, rng=rng,prune_args=prune_args,retrainindex = i)

                eval_par = {'batch_size': batch_size}
                acc = model_eval(sess, x, y, preds_adv, X_test, Y_test, args=eval_par)
                print('Test accuracy on adversarial examples: %0.4f\n' % acc)
            saver.save(sess,'./pruned_cifar_model.ckpt')
        else:
            print ("load pruned model")
            saver = tf.train.import_meta_graph('./pruned_cifar_model.ckpt.meta')
            saver.restore(sess,'./pruned_cifar_model.ckpt')
            print ("before applying gradient inhibition")
            acc = model_eval(sess, x, y, preds_adv, X_test, Y_test, args=eval_par)
            print('Test accuracy on adversarial examples by fgsm: %0.4f' % acc)
            acc = model_eval(
                    sess, x, y, preds, X_test, Y_test, args=eval_par)
            print('Test accuracy on legitimate examples: %0.4f' % acc)
        if FLAGS.do_inhibition:
            model.inhibition(sess,original_method = FLAGS.use_inhibition_original,inhibition_eps = FLAGS.inhibition_eps)

        acc = model_eval(sess, x, y, preds_adv, X_test, Y_test, args=eval_par)
        print('Test accuracy on adversarial examples by fgsm: %0.4f\n' % acc)

        acc = model_eval(
            sess, x, y, preds, X_test, Y_test, args=eval_par)
        print('Test accuracy on legitimate examples: %0.4f' % acc)
        
        '''
        some weird error occured
        Cannot feed value of shape (2, 32, 32, 3) for Tensor u'assign_timg:0', which has shape '(9, 32, 32, 3)
        we need to investigate this together
        print ("trying c&w attack")

        c_w = ElasticNetMethod(model,sess = sess)
        adv_x = c_w.generate(x)
        preds_adv = model.get_probs(adv_x)
        eval_par = {'batch_size': batch_size}
        acc = model_eval(sess, x, y, preds_adv, X_test, Y_test, args=eval_par)
        print('Test accuracy on adversarial examples: %0.4f\n' % acc)
        '''


def main(argv=None):
    cifar_tutorial(nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate,
                   clean_train=FLAGS.clean_train,
                   backprop_through_attack=FLAGS.backprop_through_attack,
                   nb_filters=FLAGS.nb_filters)


if __name__ == '__main__':
    flags.DEFINE_integer('nb_filters', 64, 'Model size multiplier')
    flags.DEFINE_integer('nb_epochs', 16, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 200, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
    flags.DEFINE_bool('clean_train', True, 'Train on clean examples')
    flags.DEFINE_bool('backprop_through_attack', False,
                      ('If True, backprop through adversarial example '
                       'construction process during adversarial training'))
    flags.DEFINE_integer('retrain_epoch',2,'Number of retrain before next pruning')
    flags.DEFINE_float('fgsm_eps',0.3,'eps for fgsm')
    flags.DEFINE_bool('use_inhibition_original',False,'true if you want to use original inhibition method. False if yo\
    u want to use my modified version')
    flags.DEFINE_integer('prune_iterations',20,'number of iteration for iterative pruning.')
    flags.DEFINE_float('retrain_lr',1e-3,'lr for retraining')
    flags.DEFINE_float('prune_factor',10,'how much percentage off. 10 as take 10 percent off')
    flags.DEFINE_float('inhibition_eps',1000,'recommend 0.1 for original, 10 for modified')
    flags.DEFINE_bool('do_inhibition',True,'set True if you want to apply gradient inhibition')
    flags.DEFINE_bool('load_pruned_model',True,'set True if you want to load pruned model')
    flags.DEFINE_bool('resume',True,'set False if you want to train from scratch')
    tf.app.run()
