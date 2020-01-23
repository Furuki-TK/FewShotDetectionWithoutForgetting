# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import os
import imp
import algorithms as alg
from dataloader import PascalVOC, FewShotDataloader

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, default='',
    help='config file with parameters of the experiment. It is assumed that all'
         ' the config file is placed on  ')
parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')

args_opt = parser.parse_args()

exp_config_file = os.path.join('.', 'config', args_opt.config + '.py')
exp_directory = os.path.join('.', 'experiments', args_opt.config)

# Load the configuration params of the experiment
print('Launching experiment: %s' % exp_config_file)
config = imp.load_source("",exp_config_file).config
config['model_dir'] = exp_directory
config['exp_dir'] = exp_directory # the place where logs, models, and other stuff will be stored
print('Loading experiment %s from file: %s' %
      (args_opt.config, exp_config_file))
print('Generated logs, snapshots, and model files will be stored on %s' %
      (config['exp_dir']))

# Set train and test datasets and the corresponding data loaders
test_split =  'test'
dataset_test = PascalVOC(phase=test_split)

data_test_opt = config['data_test_opt']
dloader = FewShotDataloader(
    dataset=dataset_test,
    nKnovel=4, # number of novel categories on each training episode.
    nKbase=10, # number of base categories.
    nExemplars=5, # num training examples per novel category
    nTestNovel=16, # num test examples for all the novel categories
    nTestBase=15, # num test examples for all the base categories
    batch_size=1,
    num_workers=0,
    epoch_size=200, # num of batches per epoch
)

algorithm = alg.FewShot(config)
if args_opt.cuda: # enable cuda
    algorithm.load_to_gpu()

# In evaluation mode we load the checkpoint with the highest novel category
# recognition accuracy on the validation set of MiniImagenet.
algorithm.load_checkpoint(epoch='*', train=False, suffix='.best')

# Run evaluation.
algorithm.evaluate(dloader)
