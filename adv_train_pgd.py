from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import scipy.sparse as sp
import matplotlib
matplotlib.use('Agg')
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PGD_attack import PGDAttack
import os


from utils import load_data, preprocess_features, preprocess_adj, construct_feed_dict
from models import GCN


C = 1. # initial  learning rate
ATTACK = True
# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_float('learning_rate', 0.002, 'Initial learning rate.')
flags.DEFINE_integer('att_steps', 15, 'Number of steps to attack.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('train_steps', 2000, 'Number of steps to train')
flags.DEFINE_bool('warm_start',False,'load saved model to start')
flags.DEFINE_bool('discrete',True,'use discret (0,1) adversarial examples to train')

flags.DEFINE_string('save_dir','adv_train_models','directory to save adversarial trained models')
if not os.path.exists(FLAGS.save_dir):
    os.makedirs(FLAGS.save_dir)


# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
total_edges = adj.data.shape[0]/2
n_node = adj.shape[0]
# Some preprocessing
features = preprocess_features(features)
# for non sparse
features = sp.coo_matrix((features[1],(features[0][:,0],features[0][:,1])),shape=features[2]).toarray()

support = preprocess_adj(adj) 
# for non sparse
support = [sp.coo_matrix((support[1],(support[0][:,0],support[0][:,1])),shape=support[2]).toarray()]
num_supports = 1
model_func = GCN

save_name = 'rob_'+FLAGS.dataset
if not os.path.exists(save_name):
   os.makedirs(save_name)

# Define placeholders
placeholders = {
    'lmd': tf.placeholder(tf.float32),
    'mu': tf.placeholder(tf.float32),
    's': [tf.placeholder(tf.float32, shape=(n_node,n_node)) for _ in range(num_supports)],
    'adj': [tf.placeholder(tf.float32, shape=(n_node,n_node)) for _ in range(num_supports)], 
    'support': [tf.placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.placeholder(tf.float32, shape=features.shape),
    'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.placeholder(tf.int32),
    'label_mask_expand': tf.placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
# for non sparse
model = model_func(placeholders, input_dim=features.shape[1], attack='PGD', logging=False)

# Initialize session
sess = tf.Session()

def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    feed_dict_val.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    outs_val = sess.run([model.attack_loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)

# Init variables
sess.run(tf.global_variables_initializer())



if FLAGS.warm_start:
    model.load(sess)
adj = adj.toarray()

nat_support = copy.deepcopy(support)
adv_support = new_adv_support = support[:]

lmd = 1
eps = total_edges * 0.05
xi = 1e-5
mu = 200
attack_label = np.load('label_'+FLAGS.dataset+'.npy')

loss_record = []
attack = PGDAttack(sess, model, features, eps, FLAGS.att_steps, mu, adj)
for n in range(FLAGS.train_steps):
    print('\n\n\n============================= iteration {}/{} =============================='.format(n+1,FLAGS.train_steps))
    print('TRAIN')
    
    
    train_label = y_train
    train_label_mask = train_mask
    
    old_adv_support = adv_support[:]
    adv_support = new_adv_support[:] 
    print('support diff:',np.sum(old_adv_support[0]-adv_support[0]))
    train_feed_dict = construct_feed_dict(features, adv_support, train_label, train_label_mask, placeholders)
    train_feed_dict.update({placeholders['support'][i]:adv_support[i] for i in range(len(adv_support))})
    train_feed_dict.update({placeholders['lmd']: lmd})
    train_feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    train_feed_dict.update({placeholders['adj'][i]: adj for i in range(num_supports)}) # feed ori adj all the time
    train_feed_dict.update({placeholders['s'][i]: np.zeros([n_node,n_node]) for i in range(num_supports)})
    train_label_mask_expand = np.tile(train_label_mask, [train_label.shape[1],1]).transpose()
    train_feed_dict.update({placeholders['label_mask_expand']: train_label_mask_expand})
    train_feed_dict.update({placeholders['mu']: 0})
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=train_feed_dict)
    loss_record.append(outs[1])
    print('[model outs] adv train acc: {}, adv train loss: {}'.format(outs[2], outs[1]))
    


    print('\n----------------------------------------------------------------------------')
    print('ATTACK')
    attack_label_mask = train_mask+test_mask 
    attack_feed_dict = construct_feed_dict(features, support, attack_label, attack_label_mask, placeholders)
    attack_feed_dict.update({placeholders['lmd']: lmd})
    attack_feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    attack_feed_dict.update({placeholders['adj'][i]: adj for i in range(num_supports)}) 
    attack_feed_dict.update({placeholders['s'][i]: np.zeros([n_node,n_node]) for i in range(num_supports)})
    attack_label_mask_expand = np.tile(attack_label_mask, [attack_label.shape[1],1]).transpose()
    attack_feed_dict.update({placeholders['label_mask_expand']: attack_label_mask_expand})
    new_adv_support = attack.perturb(attack_feed_dict, FLAGS.discrete, attack_label, attack_label_mask, FLAGS.att_steps)

    print('\n')
    train_loss, train_acc, _ = attack.evaluate(adv_support, y_train, train_mask)
    test_loss, test_acc, _ = attack.evaluate(adv_support, y_test, test_mask)
    print('[adv support] train acc: {}, train loss: {}, test acc: {}, test loss: {}'.format(train_acc, train_loss, test_acc, test_loss))
    train_loss, train_acc, _ = attack.evaluate(nat_support, y_train, train_mask)
    test_loss, test_acc, _ = attack.evaluate(nat_support, y_test, test_mask)
    print('[nat support] train acc: {}, train loss: {}, test acc: {}, test loss: {}'.format(train_acc, train_loss, test_acc, test_loss))
    
    if (n % 100 == 0 and n!=0) or n==FLAGS.train_steps-1:
       model.save(sess, save_name+'/'+save_name)

# final evaluation
new_adv_support = attack.perturb(attack_feed_dict, FLAGS.discrete, attack_label, attack_label_mask, 100)
test_cost, test_acc, test_duration = evaluate(features, new_adv_support, y_train, train_mask, placeholders)
print("Train set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
test_cost, test_acc, test_duration = evaluate(features, new_adv_support, y_val, val_mask, placeholders)
print("Validation set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
test_cost, test_acc, test_duration = evaluate(features, new_adv_support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
loss_record = np.array(loss_record)
np.save('loss_record'+FLAGS.dataset+str(eps)+'.npy',loss_record)

del sess


