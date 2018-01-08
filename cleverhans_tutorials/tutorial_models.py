"""
A pure TensorFlow implementation of a neural network. This can be
used as a drop-in replacement for a Keras model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
from cleverhans.model import Model


class MLP(Model):
    """
    An example of a bare bones multilayer perceptron (MLP) class.
    """

    def __init__(self, layers, input_shape):
        super(MLP, self).__init__()

        self.layer_names = []
        self.layers = layers
        self.input_shape = input_shape
        if isinstance(layers[-1], Softmax):
            layers[-1].name = 'probs'
            layers[-2].name = 'logits'
        else:
            layers[-1].name = 'logits'
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'name'):
                name = layer.name
            else:
                name = layer.__class__.__name__ + str(i)
            self.layer_names.append(name)

            layer.set_input_shape(input_shape)
            input_shape = layer.get_output_shape()

    def fprop(self, x, set_ref=False):
        states = []
        for layer in self.layers:
            if set_ref:
                layer.ref = x
            x = layer.fprop(x)
            assert x is not None
            states.append(x)
        states = dict(zip(self.get_layer_names(), states))
        return states
class PruneableMLP(MLP):
    def __init__(self, layers, input_shape,prune_percent=None):
        super(PruneableMLP, self).__init__(layers, input_shape)
        self.prune_percent = prune_percent
        
    def update(self):
        # update computational graph after pruning
        pass
    def prune_weight(self,weight_arr,weight_name):
        percent = self.prune_percent[weight_name]
        non_zero_weight_arr = weight_arr[weight_arr!=0]
        pcen = np.percentile(abs(non_zero_weight_arr),percent)
        print ("percentile " + str(pcen))
        under_threshold = abs(weight_arr)< pcen
        weight_arr[under_threshold] = 0
        above_threshold = weight_arr!=0
#        weight_arr[above_threshold]+=0.1*weight_arr[above_threshold]
        return [above_threshold,weight_arr]

    def apply_prune(self,sess):
        # prune layers 
          dict_nzidx = {}
          for layer in self.layers:
              if layer.weight_name in self.prune_percent.keys():
                  print ("at weight "+layer.weight_name)
                  if isinstance(layer,Conv2D):
                      weight_arr = sess.run(layer.kernels)
                  elif isinstance(layer,Linear):
                      weight_arr = sess.run(layer.W)
                  else:
                      continue
                  print ("before pruning #non zero parameters " + str(np.sum(weight_arr!=0)))
                  before = np.sum(weight_arr!=0)
                  mask,weight_arr_pruned = self.prune_weight(weight_arr,layer.weight_name)
                  after = np.sum(weight_arr_pruned!=0)
                  print ("pruned "+ str(before-after))

                  print ("after prunning #non zero parameters " + str(np.sum(weight_arr_pruned!=0)))
                  if isinstance(layer,Conv2D):

                      sess.run(layer.kernels.assign(weight_arr_pruned))
                  elif isinstance(layer,Linear):

                      sess.run(layer.W.assign(weight_arr_pruned))
                  else:
                      continue

                  dict_nzidx[layer.weight_name] = mask
          return dict_nzidx

    def apply_prune_on_grads(self,grads_and_vars,dict_nzidx):

        for key, nzidx in dict_nzidx.items():
            count = 0
            for grad, var in grads_and_vars:

                if var.name == key+":0":
                    nzidx_obj = tf.cast(tf.constant(nzidx), tf.float32)
                    grads_and_vars[count] = (tf.multiply(nzidx_obj, grad), var)
                count += 1
        return grads_and_vars
    def inhibition(self,sess,eps=0.1):
        for layer in self.layers:
            if layer.weight_name in self.prune_percent:
                print ("at %s do inhibition"% (layer.weight_name))
                weight_arr = None
                if isinstance(layer,Conv2D):
                    weight_arr = sess.run(layer.kernels)
                elif isinstance(layer,Linear):
                    weight_arr = sess.run(layer.W)
                else:
                    continue
                temp = np.zeros(weight_arr.shape)
                temp[weight_arr>0] = 1
                temp[weight_arr<0] = -1
                weight_arr += temp*0.1
                if isinstance(layer,Conv2D):
                    sess.run(layer.kernels.assign(weight_arr))
                elif isinstance(layer,Linear):

                    sess.run(layer.W.assign(weight_arr))
                else:
                    continue
            
        
    def random_pruning(self,weight_arr,weight_name):
        # double the percent                                                                                                                  
        # make the random selection                                                                                                           
        percent = 2*self.prune_percent[weight_name]

        non_zero_weight_arr = weight_arr[weight_arr!=0]
        pcen = np.percentile(abs(non_zero_weight_arr),percent)
        print ("percentile " + str(pcen))
        under_threshold = abs(weight_arr)< pcen
        shape = under_threshold.shape
        print (under_threshold.shape)
        size = np.sum(under_threshold==True)
        exclude = np.random.choice(range(size),int(size*0.5),replace=False)

        under_threshold = under_threshold.reshape(-1)
        under_threshold[exclude] = False
        under_threshold = under_threshold.reshape(shape)
        weight_arr[under_threshold] = 0
                
        above_threshold = abs(weight_arr)!=0
        return [above_threshold,weight_arr]
class Layer(object):

    def get_output_shape(self):
        return self.output_shape


class Linear(Layer):

    def __init__(self, num_hid,name):
        self.num_hid = num_hid
        self.weight_name = name
    def set_input_shape(self, input_shape):
        batch_size, dim = input_shape
        self.input_shape = [batch_size, dim]
        self.output_shape = [batch_size, self.num_hid]
        init = tf.random_normal([dim, self.num_hid], dtype=tf.float32)
        init = init / tf.sqrt(1e-7 + tf.reduce_sum(tf.square(init), axis=0,
                                                   keep_dims=True))
        self.W = tf.Variable(init,name = self.weight_name)
        self.b = tf.Variable(np.zeros((self.num_hid,)).astype('float32'))

    def fprop(self, x):
        return tf.matmul(x, self.W) + self.b


class Conv2D(Layer):

    def __init__(self, output_channels, kernel_shape, strides, padding,name=None):
        self.__dict__.update(locals())
        del self.self
        self.weight_name = name

    def set_input_shape(self, input_shape):
        batch_size, rows, cols, input_channels = input_shape
        kernel_shape = tuple(self.kernel_shape) + (input_channels,
                                                   self.output_channels)
        assert len(kernel_shape) == 4
        assert all(isinstance(e, int) for e in kernel_shape), kernel_shape
        init = tf.random_normal(kernel_shape, dtype=tf.float32)
        init = init / tf.sqrt(1e-7 + tf.reduce_sum(tf.square(init),
                                                   axis=(0, 1, 2)))
        self.kernels = tf.Variable(init,name=self.weight_name)
        self.b = tf.Variable(
            np.zeros((self.output_channels,)).astype('float32'))
        input_shape = list(input_shape)
        input_shape[0] = 1
        dummy_batch = tf.zeros(input_shape)
        dummy_output = self.fprop(dummy_batch)
        output_shape = [int(e) for e in dummy_output.get_shape()]
        output_shape[0] = 1
        self.output_shape = tuple(output_shape)

    def fprop(self, x):
        return tf.nn.conv2d(x, self.kernels, (1,) + tuple(self.strides) + (1,),
                            self.padding) + self.b


class ReLU(Layer):

    def __init__(self,name=None):
        self.weight_name = name
        pass

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def get_output_shape(self):
        return self.output_shape

    def fprop(self, x):
        return tf.nn.relu(x)


class Softmax(Layer):

    def __init__(self,name =None):
        self.weight_name = name
        pass

    def set_input_shape(self, shape):
        self.input_shape = shape
        self.output_shape = shape

    def fprop(self, x):
        return tf.nn.softmax(x)


class Flatten(Layer):

    def __init__(self,name=None):
        self.weight_name = name
        pass

    def set_input_shape(self, shape):
        self.input_shape = shape
        output_width = 1
        for factor in shape[1:]:
            output_width *= factor
        self.output_width = output_width
        self.output_shape = [None, output_width]

    def fprop(self, x):
        return tf.reshape(x, [-1, self.output_width])


def make_basic_cnn(nb_filters=64, nb_classes=10,
                   input_shape=(None, 28, 28, 1),prune_percent=None):
    layers = [Conv2D(nb_filters, (8, 8), (2, 2), "SAME",name='conv1_w'),
              ReLU(),
              Conv2D(nb_filters * 2, (6, 6), (2, 2), "VALID",name='conv2_w'),
              ReLU(),
              Conv2D(nb_filters * 2, (5, 5), (1, 1), "VALID",name='conv3_w'),
              ReLU(),
              Flatten(),
              Linear(1280,name='fc1_w'),
              Linear(1280,name='fc2_w'),
              Linear(nb_classes,name='fc3_w'),
              Softmax()]

    model = PruneableMLP(layers, input_shape,prune_percent=prune_percent)
    return model
