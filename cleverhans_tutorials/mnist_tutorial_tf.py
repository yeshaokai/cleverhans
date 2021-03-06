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

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
import logging
import time
from cleverhans.utils import pair_visual, grid_visual, AccuracyReport
from cleverhans.utils import set_log_level
from cleverhans.utils import other_classes
from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval,model_loss,model_argmax
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import SaliencyMapMethod
from cleverhans.attacks import BasicIterativeMethod,ElasticNetMethod,CarliniWagnerL2
from cleverhans_tutorials.tutorial_models import make_basic_cnn
from cleverhans.utils import AccuracyReport, set_log_level
from cleverhans.utils_tf import initialize_uninitialized_global_variables
import os

FLAGS = flags.FLAGS



def mnist_tutorial(train_start=0, train_end=60000, test_start=0,
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

    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = data_mnist(train_start=train_start,
                                                  train_end=train_end,
                                                  test_start=test_start,
                                                  test_end=test_end)

    # Use label smoothing
    assert Y_train.shape[1] == 10
    label_smooth = .1
    Y_train = Y_train.clip(label_smooth / 9., 1. - label_smooth)

    # Define input TF placeholder

    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, 10))
    
    nb_classes = 10
    source_samples = 10
    img_rows = 28
    img_cols = 28
    channels = 1

    
    model_path = "models/mnist"
    # Train an MNIST model
    train_params = {
        'nb_epochs': FLAGS.nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }

    fgsm_params = {'eps': FLAGS.fgsm_eps,
                   'clip_min': 0.,
                   'clip_max': 1.}
    rng = np.random.RandomState([2017, 8, 30])
    prune_factor = FLAGS.prune_factor
    eval_par = {'batch_size': batch_size}
    if clean_train:
        prune_percent = {'conv1_w':5,'conv2_w':5,'conv3_w':5,'conv4_w':5,'fc1_w':prune_factor,'fc2_w':prune_factor,'fc3_w':prune_factor}
        model = make_basic_cnn(nb_filters=nb_filters,prune_percent=prune_percent)
        initialize_uninitialized_global_variables(sess)
        preds = model.get_probs(x)
        saver = tf.train.Saver()
        def fgsm_combo():
            acc = model_eval(
                sess, x, y, preds, X_test, Y_test, args=eval_par)
            print('Test accuracy on legitimate examples: %0.4f\n' % acc)

            fgsm = FastGradientMethod(model, sess=sess)
            adv_x = fgsm.generate(x, **fgsm_params)
            preds_adv = model.get_probs(adv_x)
            acc = model_eval(sess, x, y, preds_adv, X_test, Y_test, args=eval_par)
            print('Test accuracy on adversarial examples generated by fgsm: %0.4f\n' % acc)

            bim = BasicIterativeMethod(model,sess = sess)
            adv_x = bim.generate(x)
            preds_adv = model.get_probs(adv_x)

            acc = model_eval(sess, x, y, preds_adv, X_test, Y_test, args=eval_par)
            print('Test accuracy on adversarial examples generated by IterativeMethod: %0.4f\n' % acc)        
        def evaluate():
            # Evaluate the accuracy of the MNIST model on legitimate test
            # examples

            acc = model_eval(
                sess, x, y, preds, X_test, Y_test, args=eval_par)
            report.clean_train_clean_eval = acc
            assert X_test.shape[0] == test_end - test_start, X_test.shape
            print('Test accuracy on legitimate examples: %0.4f' % acc)

        ckpt_name = './mnist_model.ckpt'

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
        def do_jsma():
            print('Crafting ' + str(source_samples) + ' * ' + str(nb_classes - 1) +
                  ' adversarial examples')

            # Keep track of success (adversarial example classified in target)
            results = np.zeros((nb_classes, source_samples), dtype='i')

            # Rate of perturbed features for each test set example and target class
            perturbations = np.zeros((nb_classes, source_samples), dtype='f')

            # Initialize our array for grid visualization
            grid_shape = (nb_classes, nb_classes, img_rows, img_cols, channels)
            grid_viz_data = np.zeros(grid_shape, dtype='f')

            # Instantiate a SaliencyMapMethod attack object
            jsma = SaliencyMapMethod(model, back='tf', sess=sess)
            jsma_params = {'theta': 1., 'gamma': 0.1,
                           'clip_min': 0., 'clip_max': 1.,
                           'y_target': None}

            figure = None
            # Loop over the samples we want to perturb into adversarial examples
            for sample_ind in xrange(0, source_samples):
                print('--------------------------------------')
                print('Attacking input %i/%i' % (sample_ind + 1, source_samples))
                sample = X_test[sample_ind:(sample_ind + 1)]

                # We want to find an adversarial example for each possible target class
                # (i.e. all classes that differ from the label given in the dataset)
                current_class = int(np.argmax(Y_test[sample_ind]))
                target_classes = other_classes(nb_classes, current_class)

                # For the grid visualization, keep original images along the diagonal
                grid_viz_data[current_class, current_class, :, :, :] = np.reshape(
                    sample, (img_rows, img_cols, channels))

                # Loop over all target classes
                for target in target_classes:
                    print('Generating adv. example for target class %i' % target)

                    # This call runs the Jacobian-based saliency map approach
                    one_hot_target = np.zeros((1, nb_classes), dtype=np.float32)
                    one_hot_target[0, target] = 1
                    jsma_params['y_target'] = one_hot_target
                    adv_x = jsma.generate_np(sample, **jsma_params)

                    # Check if success was achieved
                    res = int(model_argmax(sess, x, preds, adv_x) == target)

                    # Computer number of modified features
                    adv_x_reshape = adv_x.reshape(-1)
                    test_in_reshape = X_test[sample_ind].reshape(-1)
                    nb_changed = np.where(adv_x_reshape != test_in_reshape)[0].shape[0]
                    percent_perturb = float(nb_changed) / adv_x.reshape(-1).shape[0]

                    # Display the original and adversarial images side-by-side
                    if FLAGS.viz_enabled:
                        figure = pair_visual(
                            np.reshape(sample, (img_rows, img_cols)),
                            np.reshape(adv_x, (img_rows, img_cols)), figure)

                    # Add our adversarial example to our grid data
                    grid_viz_data[target, current_class, :, :, :] = np.reshape(
                        adv_x, (img_rows, img_cols, channels))

                    # Update the arrays for later analysis
                    results[target, sample_ind] = res
                    perturbations[target, sample_ind] = percent_perturb

            print('--------------------------------------')

            # Compute the number of adversarial examples that were successfully found
            nb_targets_tried = ((nb_classes - 1) * source_samples)
            succ_rate = float(np.sum(results)) / nb_targets_tried
            print('Avg. rate of successful adv. examples {0:.4f}'.format(succ_rate))
            report.clean_train_adv_eval = 1. - succ_rate

            # Compute the average distortion introduced by the algorithm
            percent_perturbed = np.mean(perturbations)
            print('Avg. rate of perturbed features {0:.4f}'.format(percent_perturbed))

            # Compute the average distortion introduced for successful samples only
            percent_perturb_succ = np.mean(perturbations * (results == 1))
            print('Avg. rate of perturbed features for successful '
                  'adversarial examples {0:.4f}'.format(percent_perturb_succ))
            if FLAGS.viz_enabled:
                import matplotlib.pyplot as plt
                plt.close(figure)
                _ = grid_visual(grid_viz_data)

            return report
        def do_cw():
            nb_adv_per_sample = str(nb_classes - 1) if FLAGS.targeted else '1'
            print('Crafting ' + str(source_samples) + ' * ' + nb_adv_per_sample +
                  ' adversarial examples')
            print("This could take some time ...")

            # Instantiate a CW attack object
            cw = CarliniWagnerL2(model, back='tf', sess=sess)

            if FLAGS.viz_enabled:
                assert source_samples == nb_classes
                idxs = [np.where(np.argmax(Y_test, axis=1) == i)[0][0]
                        for i in range(nb_classes)]
            if FLAGS.targeted:
                if FLAGS.viz_enabled:
                    # Initialize our array for grid visualization
                    grid_shape = (nb_classes, nb_classes, img_rows, img_cols, channels)
                    grid_viz_data = np.zeros(grid_shape, dtype='f')

                    adv_inputs = np.array(
                        [[instance] * nb_classes for instance in X_test[idxs]],
                        dtype=np.float32)
                else:
                    adv_inputs = np.array(
                        [[instance] * nb_classes for
                         instance in X_test[:source_samples]], dtype=np.float32)

                one_hot = np.zeros((nb_classes, nb_classes))
                one_hot[np.arange(nb_classes), np.arange(nb_classes)] = 1

                adv_inputs = adv_inputs.reshape(
                        (source_samples * nb_classes, img_rows, img_cols, 1))
                adv_ys = np.array([one_hot] * source_samples,
                                      dtype=np.float32).reshape((source_samples *
                                                                 nb_classes, nb_classes))
                yname = "y_target"
            else:
                if FLAGS.viz_enabled:
                    # Initialize our array for grid visualization
                    grid_shape = (nb_classes, 2, img_rows, img_cols, channels)
                    grid_viz_data = np.zeros(grid_shape, dtype='f')

                    adv_inputs = X_test[idxs]
                else:
                    adv_inputs = X_test[:source_samples]

                adv_ys = None
                yname = "y"

            cw_params = {'binary_search_steps': 1,
                         yname: adv_ys,
                         'max_iterations': FLAGS.attack_iterations,
                         'learning_rate': 0.1,
                                 'batch_size': source_samples * nb_classes if
                         FLAGS.targeted else source_samples,
                         'initial_const': 10}

            adv = cw.generate_np(adv_inputs,
                                         **cw_params)


            if FLAGS.targeted:
                adv_accuracy = model_eval(
                    sess, x, y, preds, adv, adv_ys, args=eval_par)
            else:
                if FLAGS.viz_enabled:
                    adv_accuracy = 1 - \
                                   model_eval(sess, x, y, preds, adv, Y_test[
                                       idxs], args=eval_par)
                else:
                    adv_accuracy = 1 - \
                                   model_eval(sess, x, y, preds, adv, Y_test[
                                       :source_samples], args=eval_par)

            if FLAGS.viz_enabled:
                for j in range(nb_classes):
                    if FLAGS.targeted:
                        for i in range(nb_classes):
                            grid_viz_data[i, j] = adv[i * nb_classes + j]
                    else:
                        grid_viz_data[j, 0] = adv_inputs[j]
                        grid_viz_data[j, 1] = adv[j]

                print(grid_viz_data.shape)

            print('--------------------------------------')

            # Compute the number of adversarial examples that were successfully found
            print('Avg. rate of successful adv. examples {0:.4f}'.format(adv_accuracy))
            report.clean_train_adv_eval = 1. - adv_accuracy

            # Compute the average distortion introduced by the algorithm
            percent_perturbed = np.mean(np.sum((adv - adv_inputs)**2,
                                                                               axis=(1, 2, 3))**.5)
            print('Avg. L_2 norm of perturbations {0:.4f}'.format(percent_perturbed))
                                             # Close TF session
        #            sess.close()

            # Finally, block & display a grid of all the adversarial examples
            if FLAGS.viz_enabled:
                import matplotlib.pyplot as plt
                _ = grid_visual(grid_viz_data)

            return report
        print ("before pruning and gradient inhibition\n")
        fgsm_combo()
        do_cw()
        do_jsma()
        preds = model.get_probs(x)
        loss = model_loss(y,preds)
        if not FLAGS.load_pruned_model:
            print ("start iterative pruning")
            for i in range(FLAGS.prune_iterations):
                print ("iterative %d"  % (i))
                start = time.time()
                dict_nzidx = model.apply_prune(sess)
                trainer = tf.train.AdamOptimizer(learning_rate)
                grads = trainer.compute_gradients(loss)
                grads = model.apply_prune_on_grads(grads,dict_nzidx)
                end = time.time()
                print ('until grad compute elpased %f' % (end-start))
                prune_args = {'trainer':trainer,'grads':grads}
                train_params = {
                    'nb_epochs':FLAGS.retrain_epoch,
                    'batch_size': batch_size,
                    'learning_rate': FLAGS.retrain_lr
                    }
                start = time.time()
                model_train(sess, x, y, preds, X_train, Y_train, evaluate=evaluate,
                            args=train_params, rng=rng,prune_args=prune_args,retrainindex = i)
                end = time.time()
                print ('model_train function takes %f' % (end-start))
                eval_par = {'batch_size': batch_size}
                acc = model_eval(sess, x, y, preds_adv, X_test, Y_test, args=eval_par)
                print('Test accuracy on adversarial examples: %0.4f\n' % acc)
            saver.save(sess,'./pruned_mnist_model.ckpt')
        else:
            print ("loading pruned model")
            saver = tf.train.import_meta_graph('./pruned_mnist_model.ckpt.meta')
            saver.restore(sess,'./pruned_mnist_model.ckpt')
            print ("before  applying gradient inhibition")
            fgsm_combo()
            print("before gradient inhibition, doing c&w")
            do_cw()
            do_jsma()
        if FLAGS.do_inhibition:
            model.inhibition(sess,original_method = FLAGS.use_inhibition_original,inhibition_eps = FLAGS.inhibition_eps)


        fgsm_combo()
        do_cw()
        do_jsma()





def main(argv=None):
    mnist_tutorial(nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
                   learning_rate=FLAGS.learning_rate,
                   clean_train=FLAGS.clean_train,
                   backprop_through_attack=FLAGS.backprop_through_attack,
                   nb_filters=FLAGS.nb_filters)


if __name__ == '__main__':
    flags.DEFINE_integer('nb_filters', 32, 'Model size multiplier')
    flags.DEFINE_integer('nb_epochs', 12, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 1024, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
    flags.DEFINE_bool('clean_train', True, 'Train on clean examples')
    flags.DEFINE_bool('backprop_through_attack', False,
                      ('If True, backprop through adversarial example '
                       'construction process during adversarial training'))
    flags.DEFINE_integer('retrain_epoch',2,'Number of retrain before next pruning')
    flags.DEFINE_float('fgsm_eps',0.3,'eps for fgsm')
    flags.DEFINE_bool('use_inhibition_original',False,'true if you want to use original inhibition method. False if you want to use my modified version')
    flags.DEFINE_integer('prune_iterations',20,'number of iteration for iterative pruning.')
    flags.DEFINE_float('retrain_lr',1e-3,'lr for retraining')
    flags.DEFINE_float('prune_factor',10,'how much percentage off. 10 as take 10 percent off')
    flags.DEFINE_float('inhibition_eps',100,'recommend 0.1 for original, 20 for modified')
    flags.DEFINE_bool('do_inhibition',True,'set True if you want to apply gradient inhibition')
    flags.DEFINE_bool('load_pruned_model',True,'set True if you want to load from the pruned model')
    flags.DEFINE_bool('resume',True,'set False if you want to train from scratch')
    flags.DEFINE_boolean('attack_iterations', 50000,
                         'Number of iterations to run attack; 1000 is good')
    flags.DEFINE_boolean('targeted', True,
                         'Run the tutorial in targeted mode?')
    flags.DEFINE_boolean('viz_enabled', False, 'Visualize adversarial ex.')
    tf.app.run()
