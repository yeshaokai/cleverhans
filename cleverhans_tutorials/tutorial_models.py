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
from cleverhans.utils_tf import initialize_uninitialized_global_variables

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
    def test_mode(self):
        for layer in self.layers:
            if isinstance(layer,BN):
                layer.test_mode()
    def vis_weights(self,sess,name ='fc3_w',title='no title'):
        import pandas as pd
        import matplotlib.pyplot as plt
        import matplotlib
        import operator
        #matplotlib.use('Agg')
        array = []
        weight_arr = None
        for layer in self.layers:
            if isinstance(layer,Conv2D):
                weight_arr = sess.run(layer.kernels)
            elif isinstance(layer,Linear):
                weight_arr = sess.run(layer.W)
            else:
                continue
            #nonzero = weight_arr[weight_arr!=0]
            weight_arr = weight_arr.reshape(-1)
            nonzero = list(weight_arr)
            #print (nonzero.shape)
            array.append(nonzero)
        array = reduce(operator.add,array)
        data = pd.DataFrame(array)
        data.hist(bins=200,log = True)
        plt.title(title)
        plt.ylabel('occurences')
        plt.show()
                
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

    def inhibition(self,sess,original_method = False, inhibition_eps = 20):
        for layer in self.layers:
            if layer.weight_name in self.prune_percent and 'conv' not in layer.weight_name:
               
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
                if original_method:
                    print ("at %s do inhibition"% (layer.weight_name))
                    weight_arr += inhibition_eps*temp
                else:
                  
                    if 'fc1_w' in layer.weight_name or 'fc2_w' in layer.weight_name:
                        continue
                    print ("using modified gradient inhibition")
                    print ("at %s do inhibition"% (layer.weight_name))
                    weight_arr += weight_arr*inhibition_eps
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
    '''
    wd:  weight decay

    '''
    def __init__(self, num_hid,name,wd=0.01):
        self.num_hid = num_hid
        self.weight_name = name
        self.wd = wd
    def set_input_shape(self, input_shape):


        batch_size, dim = input_shape
        self.input_shape = [batch_size, dim]
        self.output_shape = [batch_size, self.num_hid]
        init = tf.random_normal([dim, self.num_hid], dtype=tf.float32)
        init = init / tf.sqrt(1e-7 + tf.reduce_sum(tf.square(init), axis=0,
                                                   keep_dims=True))
        #self.W = tf.Variable(init,name = self.weight_name)
        regularizer = tf.contrib.layers.l2_regularizer(self.wd)
        self.W = tf.get_variable(self.weight_name,shape=[dim,self.num_hid],regularizer = regularizer)
        self.b = tf.Variable(np.zeros((self.num_hid,)).astype('float32'))

    def fprop(self, x):
        return tf.matmul(x, self.W) + self.b

class BN(Layer):
    def __init__(self,name = None):
        self.weight_name = name
        self.batch = None
        self.is_training = True
    def set_input_shape(self,input_shape):
        self.input_shape = input_shape        
        self.output_shape = input_shape
    def test_mode(self):
        self.is_training = False
    def train_mode(self):
        self.is_training = True
    def fprop(self,input_layer):
        with tf.variable_scope('%s'%self.weight_name,reuse = tf.AUTO_REUSE):
            return tf.contrib.layers.batch_norm(input_layer, is_training = self.is_training)
        
def bn_relu_conv_layer(input_layer, filter_shape, stride):
    '''                                                                                                                               
    A helper function to batch normalize, relu and conv the input layer sequentially                                                  
    :param input_layer: 4D tensor                                                                                                     
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]                                             
    :param stride: stride size for conv                                                                                               
    :return: 4D tensor. Y = conv(Relu(batch_normalize(X)))                                                                            
    '''

    in_channel = input_layer.get_shape().as_list()[-1]
    
    bn_layer = batch_normalization_layer(input_layer, in_channel)
    relu_layer = tf.nn.relu(bn_layer)
    
    filter = tf.get_variable(name='conv',shape=filter_shape)#create_variables(name='conv', shape=filter_shape)

    conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    return conv_layer
def conv_bn_relu_layer(input_layer, filter_shape, stride):
    '''                                                                                                                               
    A helper function to conv, batch normalize and relu the input tensor sequentially                                                 
    :param input_layer: 4D tensor                                                                                                     
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]                                             
    :param stride: stride size for conv                                                                                               
    :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))                                                                            
    '''

    out_channel = filter_shape[-1]
    filter = create_variables(name='conv', shape=filter_shape)
        
    conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    bn_layer = batch_normalization_layer(conv_layer, out_channel)
        
    output = tf.nn.relu(bn_layer)
    return output
class residual_block(Layer):
    def __init__(self,input_channel,output_channel,first_block=False,postfix='yo',name = None):
        self.postfix = postfix
        self.output_channel = output_channel
        self.first_block = first_block
        self.input_channel = input_channel
        self.kernel_shape = (3,3)
        self.increase_dim = None
        self.tride = None
        self.weight_name = name
    def set_input_shape(self,input_shape):
        input_layer = tf.zeros(input_shape)
        if self.input_channel*2 == self.output_channel:
            self.increase_dim = True
            self.stride = 2
        elif self.input_channel == self.output_channel:
            self.increase_dim = False
            self.stride = 1
        else:
            raise ValueError('Output and input channel does not match in residual blocks!!!')

        with tf.variable_scope('dummy'):
            if self.first_block:
                filter = tf.get_variable(name='%s'%self.postfix,shape=[3, 3, self.input_channel, self.output_channel])
                conv1 = tf.nn.conv2d(input_layer, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
            else:
                conv1 = bn_relu_conv_layer(input_layer, [3, 3, self.input_channel, self.output_channel], self.stride)                
        with tf.variable_scope('dummy_2'):
            conv2 = bn_relu_conv_layer(conv1, [3, 3, self.output_channel, self.output_channel], 1)
        if self.increase_dim is True:
            pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                          strides=[1, 2, 2, 1], padding='VALID')
            padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [self.input_channel // 2,
                                                                          self.input_channel // 2]])
        else:
            padded_input = input_layer
        output = conv2 + padded_input
        self.output_shape = output.get_shape()
        
    def fprop(self,input_layer):
        with tf.variable_scope('conv1_in_block_%s'%self.postfix,reuse=tf.AUTO_REUSE):
            if self.first_block:
                filter = tf.get_variable(name='conv1',shape=[3, 3, self.input_channel, self.output_channel])
                conv1 = tf.nn.conv2d(input_layer, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
            else:
                conv1 = bn_relu_conv_layer(input_layer, [3, 3, self.input_channel, self.output_channel], self.stride)                
        with tf.variable_scope('conv2_in_block_%s'%self.postfix,reuse=tf.AUTO_REUSE):
            conv2 = bn_relu_conv_layer(conv1, [3, 3, self.output_channel, self.output_channel], 1)
        if self.increase_dim is True:
            pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                          strides=[1, 2, 2, 1], padding='VALID')
            padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [self.input_channel // 2,
                                                                          self.input_channel // 2]])
        else:
            padded_input = input_layer
        output = conv2 + padded_input        
        return output
                        
        
class Conv2D(Layer):

    def __init__(self, output_channels, kernel_shape, strides, padding,name=None):
        self.__dict__.update(locals())
        del self.self
        self.weight_name = name
        self.kernel_shape = kernel_shape
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
        conv_layer =  tf.nn.conv2d(x, self.kernels, (1,) + tuple(self.strides) + (1,),
                            self.padding) + self.b
        return conv_layer


def batch_normalization_layer(input_layer, dimension,scope = None):
    '''                                                                                                                               
    Helper function to do batch normalziation                                                                                         
    :param input_layer: 4D tensor                                                                                                     
    :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor                                               
    :return: the 4D tensor after being normalized                                                                                     
    '''
    return tf.contrib.layers.batch_norm(input_layer)

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

class Global_Pool(Layer):
    def __init__(self,name = None):
        self.weight_name = name
    def set_input_shape(self,shape):
        self.output_shape = (shape[0],shape[-1])
    def fprop(self,x):
        return tf.reduce_mean(x,[1,2])
class MaxPool(Layer):
    def __init__(self,strides=2,padding='VALID',name=None):
        self.weight_name = name
        self.strides = strides
        self.padding = padding

    def set_input_shape(self,shape):
        self.input_shape = shape
        dummy_batch = tf.zeros(shape)
        dummy_output = self.fprop(dummy_batch)
        output_shape = [int(e) for e in dummy_output.get_shape()]
        output_shape[0] = 1
        self.output_shape = tuple(output_shape)
    def fprop(self,x):
        return tf.layers.max_pooling2d(x,pool_size=[2,2],strides=self.strides,padding=self.padding)
class Softmax(Layer):

    def __init__(self,name =None):
        self.weight_name = name

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

def make_compare_cnn(nb_filters=32, nb_classes=10,
                   input_shape=(None, 28, 28, 1),prune_percent=None):
    with tf.variable_scope('compare'):
        layers = [Conv2D(nb_filters, (3, 3), (1, 1), "SAME",name='conv1_w'),
              ReLU(),
              Conv2D(nb_filters, (3, 3), (1, 1), "SAME",name='conv2_w'),
              ReLU(),
              MaxPool(),
              Conv2D(nb_filters * 2, (3, 3), (1, 1), "SAME",name='conv3_w'),
              ReLU(),
              MaxPool(),
              Flatten(),
              Linear(200,name='fc1_w'),
              ReLU(),
              Linear(200,name='fc2_w'),
              ReLU(),
              Linear(nb_classes,name='fc3_w'),
              Softmax()]
        model = PruneableMLP(layers, input_shape,prune_percent=prune_percent)
        return model
def make_basic_cnn(nb_filters=32, nb_classes=10,
                   input_shape=(None, 28, 28, 1),prune_percent=None):
    layers = [Conv2D(nb_filters, (3, 3), (1, 1), "SAME",name='conv1_w'),
              ReLU(),
              Conv2D(nb_filters, (3, 3), (1, 1), "SAME",name='conv2_w'),
              ReLU(),
              MaxPool(),
              Conv2D(nb_filters * 2, (3, 3), (1, 1), "SAME",name='conv3_w'),
              ReLU(),
              Conv2D(nb_filters * 2, (3, 3), (1, 1), "SAME",name='conv4_w'),
              ReLU(),
              MaxPool(),
              Flatten(),
              Linear(200,name='fc1_w'),
              ReLU(),
              Linear(200,name='fc2_w'),
              ReLU(),
              Linear(nb_classes,name='fc3_w'),
              Softmax()]
    model = PruneableMLP(layers, input_shape,prune_percent=prune_percent)
    return model
def make_strong_cnn(nb_filters=64, nb_classes=10,
                   input_shape=(None, 32, 32, 3),prune_percent=None):
    layers = [Conv2D(nb_filters, (3, 3), (1, 1), "SAME",name='conv1_w'),
              ReLU(),
              Conv2D(nb_filters, (3, 3), (1, 1), "SAME",name='conv2_w'),
              ReLU(),
              MaxPool(),
              Conv2D(nb_filters * 2, (3, 3), (1, 1), "SAME",name='conv3_w'),
              ReLU(),
              Conv2D(nb_filters * 2, (3, 3), (1, 1), "SAME",name='conv4_w'),
              ReLU(),
              MaxPool(),
              Flatten(),
              Linear(256,name='fc1_w'),
              ReLU(),
              Linear(256,name='fc2_w'),
              ReLU(),
              Linear(nb_classes,name='fc3_w'),
              Softmax()]

    model = PruneableMLP(layers, input_shape,prune_percent=prune_percent)
    return model
def make_resnet(x,n,input_shape,reuse,prune_percent):
    layers = []
    #with tf.variable_scope('conv0', reuse=reuse):

        # replace conv0 as conv, bn, relu
    with tf.variable_scope('initial',reuse = tf.AUTO_REUSE):
        layers.append(Conv2D(16,(3,3),(1,1),'SAME'))
        layers.append(BN(16,16,scope='bn1'))
        layers.append(ReLU())
    reuse = None
    for i in range(n):
        if i >0:
            reuse = True
        else:
            reuse = False
        with tf.variable_scope('conv1_%d' %i, reuse=tf.AUTO_REUSE):
            if i == 0:
                conv1 = residual_block(16,16, first_block=True,postfix='conv1_%d' %i)
            else:
                conv1 = residual_block(16,16,postfix='conv1_%d' %i)
            layers.append(conv1)
    out_channel = 0
    for i in range(n):
        with tf.variable_scope('conv2_%d' %i, reuse=tf.AUTO_REUSE):
            if i == 0:
                out_channel = 16
            else:
                out_channel = 32
            conv2 = residual_block(out_channel,32,postfix='conv2_%d' %i)
            layers.append(conv2)
    
    for i in range(n):
        with tf.variable_scope('conv3_%d' %i, reuse=tf.AUTO_REUSE):
            if i == 0:
                out_channel = 32
            else:
                out_channel = 64
            conv3 = residual_block(out_channel,64,postfix='conv3_%d' %i)
            layers.append(conv3)
            
    with tf.variable_scope('fc', reuse=tf.AUTO_REUSE):
        in_channel = layers[-1].output_channel            

        layers.append(BN(64,64,scope='bn2'))
        layers.append(ReLU())
        layers.append(Global_Pool())
        layers.append(Linear(10,name='fc1_w'))
        layers.append(Softmax())
        print ("build the graph")
        model = PruneableMLP(layers,input_shape,prune_percent=prune_percent)

    return model
