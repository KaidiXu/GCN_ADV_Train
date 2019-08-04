from layers import *
from metrics import *
import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS



class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._attack_loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError
    
    def _attack_loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None, path=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        if not path:
            save_path = saver.save(sess, "tmp1/%s.ckpt" % self.name)
        else:
            save_path = saver.save(sess, path) 
        print("Model saved in file: %s" % save_path)
        
    def load_original(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp1/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)
        
    def load(self, path, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars) 
        save_path = tf.train.latest_checkpoint(path)
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN(Model):
    def __init__(self, placeholders, input_dim, attack=None, **kwargs):
        super(GCN, self).__init__(**kwargs)
        print('attack method:',attack)
        # if attack is False, placeholders['support'] feeds in normalized pre-processed adjacent matrix, 
        # if attack is True, placeholders['adj'] feeds in raw adjacent matrix and placeholdder['s'] feeds in attack placeholders
        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders
        lmd = placeholders['lmd']
        self.attack = attack 

        if self.attack:
            mu = placeholders['mu']
            
            # the length of A list, in fact, self.num_support is always 1
            self.num_supports = len(placeholders['adj'])
            # original adjacent matrix A
            self.A = placeholders['adj']
            self.mask = [tf.constant(np.triu(np.ones([self.A[0].get_shape()[0].value]*2, dtype = np.float32),1))]
             
            self.C = [1 - 2 * self.A[i] - tf.eye(self.A[i].get_shape().as_list()[0], self.A[i].get_shape().as_list()[1]) for i in range(self.num_supports)] 
            # placeholder for adding edges
            self.upper_S_0 = placeholders['s'] 
            # a strict upper triangular matrix to ensure only N(N-1)/2 trainable variables
            # here use matrix_band_part to ensure a stricly upper triangular matrix     
            self.upper_S_real = [tf.matrix_band_part(self.upper_S_0[i],0,-1)-tf.matrix_band_part(self.upper_S_0[i],0,0) for i in range(self.num_supports)] 
            # modified_A is the new adjacent matrix
            self.upper_S_real2 = [self.upper_S_real[i] + tf.transpose(self.upper_S_real[i]) for i in range(self.num_supports)]
            self.modified_A = [self.A[i] + tf.multiply(self.upper_S_real2[i], self.C[i]) for i in range(self.num_supports)]
            """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""   
            self.hat_A = [tf.cast(self.modified_A[i] + tf.eye(self.modified_A[i].get_shape().as_list()[0], self.modified_A[i].get_shape().as_list()[1]),dtype='float32') for i in range(self.num_supports)] 
            
            # get degree by row sum
            self.rowsum = tf.reduce_sum(self.hat_A[0],axis=1) 
            self.d_sqrt = tf.sqrt(self.rowsum) # square root
            self.d_sqrt_inv = tf.math.reciprocal(self.d_sqrt) # reciprocal
            
            self.support_real = tf.multiply(tf.transpose(tf.multiply(self.hat_A[0],self.d_sqrt_inv)),self.d_sqrt_inv)
            # this self.support is a list of \tilde{A} in the paper
            # replace the 'support' in the placeholders dictionary
            self.placeholders['support'] = [self.support_real] 
            
            self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
            self.build()
            
            
            
            # proximal gradient algorithm
            if self.attack == 'PGD':
                self.Sgrad = tf.gradients(self.attack_loss, self.upper_S_real[0])
                self.a = self.upper_S_real[0] + mu * self.Sgrad * lmd * self.mask
            elif self.attack == 'CW':
                label = placeholders['labels'] 
                real = tf.reduce_sum(label * self.outputs,1)
                label_mask_expand = placeholders['label_mask_expand']
                other = tf.reduce_max((1 - label) * label_mask_expand * self.outputs - label * 10000,1)
                self.loss1 = tf.maximum(0.0, (real-other+50)*label_mask_expand[:,0])
                self.loss2 = tf.reduce_sum(self.loss1) 
                self.Sgrad = tf.gradients(self.loss2, self.upper_S_real[0])
                self.a = self.upper_S_real[0] - mu * self.Sgrad * lmd * self.mask
            elif self.attack == 'minmax':
                self.w = placeholders['w']
                label = placeholders['labels'] 
                self.real = tf.reduce_sum(label * self.outputs,1)
                label_mask_expand = placeholders['label_mask_expand']
                self.other = tf.reduce_max((1 - label) * label_mask_expand * self.outputs - label * 10000,1)
                self.loss1 = self.w * tf.maximum(0.0, self.real-self.other+0.)
                self.loss2 = tf.reduce_sum(self.loss1) 
                self.Sgrad = tf.gradients(self.loss2, self.upper_S_real[0])
                self.a = self.upper_S_real[0] - mu * self.Sgrad * self.mask
            else:
                raise NotImplementedError
            
            
        else:
            self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
            self.build()
        
        
    def _attack_loss(self):
        # Cross entropy error
        self.attack_loss = masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=False,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)

        
            
