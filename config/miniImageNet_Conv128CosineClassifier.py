config = {}
# set the parameters related to the training and testing set

nKbase = 64

data_train_opt = {}
data_train_opt['nKnovel'] = 0
data_train_opt['nKbase'] = nKbase
data_train_opt['nExemplars'] = 0
data_train_opt['nTestNovel'] = 0
data_train_opt['nTestBase'] = 32
data_train_opt['batch_size'] = 8
data_train_opt['epoch_size'] = data_train_opt['batch_size'] * 1000

data_test_opt = {}
data_test_opt['nKnovel'] = 5
data_test_opt['nKbase'] = nKbase
data_test_opt['nExemplars'] = 1
data_test_opt['nTestNovel'] = 15 * data_test_opt['nKnovel']
data_test_opt['nTestBase'] = 15 * data_test_opt['nKnovel']
data_test_opt['batch_size'] = 1
data_test_opt['epoch_size'] = 2000

config['data_train_opt'] = data_train_opt
config['data_test_opt'] = data_test_opt

config['max_num_epochs'] = 60

networks = {}
net_optionsF = {'userelu': False, 'in_planes':3, 'out_planes':[64,64,128,128], 'num_stages':4}
net_optim_paramsF = {'optim_type': 'sgd', 'lr': 0.1, 'momentum':0.9, 'weight_decay': 5e-4, 'nesterov': True, 'LUT_lr':[(20, 0.1),(40, 0.006),(50, 0.0012),(60, 0.00024)]}
networks['feat_model'] = {'def_file': 'architectures/ConvNet.py', 'pretrained': None, 'opt': net_optionsF, 'optim_params': net_optim_paramsF}

net_optim_paramsC = {'optim_type': 'sgd', 'lr': 0.1, 'momentum':0.9, 'weight_decay': 5e-4, 'nesterov': True, 'LUT_lr':[(20, 0.1),(40, 0.006),(50, 0.0012),(60, 0.00024)]}
net_optionsC = {'classifier_type': 'cosine', 'weight_generator_type': 'none', 'nKall': nKbase, 'nFeat':128*5*5, 'scale_cls': 10}
networks['classifier'] = {'def_file': 'architectures/ClassifierWithFewShotGenerationModule.py', 'pretrained': None, 'opt': net_optionsC, 'optim_params': net_optim_paramsC}

config['networks'] = networks

criterions = {}
criterions['loss'] = {'ctype':'CrossEntropyLoss', 'opt':None}
config['criterions'] = criterions

config['algorithm_type'] = 'FewShot'
