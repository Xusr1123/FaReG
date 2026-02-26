import os, sys
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import shutil
import shap
from scipy.spatial.distance import pdist
# from audtorch.metrics.functional import pearsonr

import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split

sys.path.append("../")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from third_party.utils import *
from Group_fairness.wsc import *
from Group_fairness.cp import Marginal_Fairness, FaReG_Fairness
from Group_fairness.black_box import Blackbox, FaReG_Train
from Group_fairness.networks import ClassNNet, FaReG
from Group_fairness.evals import eval_psets, eval_by_att_mc_1, eval_by_att_mc_2 
from experiment.data_gen import data_model

import warnings
warnings.filterwarnings('ignore')

#########################
# Experiment parameters #
#########################

# Default parameters
n_train_calib = 1000
lr = 0.0001
n_epoch = 500
delta0 = 0.1 
delta1 = 0.9
ttest_delta = 0
conditional_ = False
perc_1 = 0.5
perc_2 = 0.5

# Fixed experiment parameters
num_workers = 0
batch_size = 250
beta = 0 
alpha = 0.1 # nominal miscoverage level
n_test = 500

# WSC plus
g_delta = 0.5
M = 1000


###############
# Output file #
###############
outdir = "results/"
os.makedirs(outdir, exist_ok=True)
outfile_name = "ndata"+str(n_train_calib) + "_lr" + str(lr) + "_delta1" + str(delta1) + \
               "_ttestdelta" + str(ttest_delta) + "_conditional" + str(conditional_)
              

modeldir = "models/baseModel/" + outfile_name


p = 10 # dimension of the feature for simulated data
K = 6 # number of classes
idx_list = [p-1, p-2, p-3, p-4] # list of attributes to investigate over


group_perc_1 = [1-perc_1,perc_1] # easy-to-classify, hard-to-classify
group_perc_2 = [1-perc_2,perc_2] # easy-to-classify, hard-to-classify


#########################
# Auxiliary functions #
#########################


def criterion(outputs, inputs, targets):
    targets = targets.to(torch.long)
    return Loss(outputs, targets)

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        # Make CuDNN Determinist
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed) 
    
def complete_df(df, method = None, seed = -1, is_marg_results = False):
    # assert method is not None, 'need to input the method name!'
    df["n_data"] = n_train_calib
    df["seed"] = seed
    df["lr"] = lr
    df["batch_size"] = batch_size
    df['alpha'] = alpha
    df['delta1'] = delta1
    df['labcond'] = conditional_
    df['method'] = method
    df['perc_1'] = perc_1
    if beta>0:
        df['beta_threshold'] = beta*len(X_calib)
    if is_marg_results:
        df['n_count'] = len(y_test)
        df['attribute_idx'] = -1
        df['attribute_level'] = -1

    return df

def compute_wsc(X, y, S, delta, M, random_state=2025):
    cov_list = mp_wsc_plus(X, y, S, delta=delta, M=M, random_state=random_state)

    return np.min(cov_list)

#################
# Download/Simulate Data #
#################

seed = 0
set_seed(seed)

data_sampler = data_model(K = K, p = p, delta0 = delta0, delta1 = delta1, 
                        group_perc_1 = group_perc_1, group_perc_2 = group_perc_2, seed = seed)           # Data generating model, group = [easy, hard]


X_full = data_sampler.sample_X(n_train_calib)
y_full = data_sampler.sample_Y(X_full)

X_train, X_calib, y_train, y_calib = train_test_split(X_full, y_full, train_size = 0.5, random_state = seed)

X_test = data_sampler.sample_X(n_test)
y_test = data_sampler.sample_Y(X_test)

# convert to dataset and dataloader
train_dataset = data_utils.TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
calib_dataset = data_utils.TensorDataset(torch.Tensor(X_calib), torch.Tensor(y_calib))
test_dataset = data_utils.TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
calib_loader = torch.utils.data.DataLoader(calib_dataset, batch_size=batch_size, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=n_test, num_workers=num_workers, shuffle= False)



################
# Train models #
################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Is CUDA available? {}".format(torch.cuda.is_available()))


# Define the model parameters
net = ClassNNet(num_features = p, num_classes = K, device = device, use_dropout=False)

Loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=lr)
# Training the model
bbox_mc = Blackbox(net, device, train_loader, batch_size=batch_size, max_epoch=n_epoch,
                learning_rate=lr, val_loader=calib_loader, criterion=criterion, optimizer=optimizer, verbose = True)

bbox_mc_stats = bbox_mc.full_train(save_dir = modeldir + "_seed" + str(seed))
bbox_mc.net.load_state_dict(torch.load(modeldir + "_seed" + str(seed))['model_state'])
print('Test acc of bbox_mc: ', bbox_mc.get_acc(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()))

################
#   Inference  #
################

results_full = pd.DataFrame()

# Marginal
Marginal_method = Marginal_Fairness(alpha = alpha, random_state = seed)
C_sets_marginal = Marginal_method.multiclass_classification(X_test,
                                                            X_calib,
                                                            y_calib,
                                                            left_tail = False,
                                                            bbox_mc = bbox_mc,
                                                            conditional = conditional_)


# marginal results, on all combinations
results_marg = eval_psets(C_sets_marginal, y_test)
results_marg = complete_df(results_marg, method = 'Marginal', seed = seed, is_marg_results = True)
results_cond_1 = eval_by_att_mc_1(X_test, y_test, idx_list, C_sets_marginal)
results_cond_1 = complete_df(results_cond_1, method = 'Marginal', seed = seed)
results_cond_2 = eval_by_att_mc_2(X_test, y_test, idx_list, C_sets_marginal)
results_cond_2 = complete_df(results_cond_2, method = 'Marginal', seed = seed)
results_full = pd.concat([results_marg, results_cond_1, results_cond_2])


C_sets_marginal_calib = Marginal_method.multiclass_classification(X_calib,
                                                                    X_calib,
                                                                    y_calib,
                                                                    left_tail = False,
                                                                    bbox_mc = bbox_mc,
                                                                    conditional = conditional_)

################
# Train FaReG  #
################
lr2 = 0.001
batch2 = 500
epoch2 = 2000
delta = 0.3
n_sample = 20

covs_calib = [1 if y_calib[i] in C_sets_marginal_calib[i] else 0 for i in range(len(y_calib))]

X_calib_train, X_calib_val, covs_calib_train, covs_calib_val = train_test_split(X_calib, covs_calib, train_size = 0.5, random_state = seed)

# convert to dataset and dataloader
fareg_train_dataset = data_utils.TensorDataset(torch.Tensor(X_calib_train[:,-4:]), torch.Tensor(covs_calib_train))
fareg_val_dataset = data_utils.TensorDataset(torch.Tensor(X_calib_val[:,-4:]), torch.Tensor(covs_calib_val))

fareg_train_loader = torch.utils.data.DataLoader(fareg_train_dataset, batch_size=batch2, num_workers=num_workers)
fareg_val_loader = torch.utils.data.DataLoader(fareg_val_dataset, batch_size=batch2, num_workers=num_workers)


# Define the model parameters
fareg = FaReG(num_features = len(idx_list), delta = delta, device = device)

fareg_optimizer = optim.Adam(fareg.parameters(), lr=lr2)
# Training the model
fareg_train = FaReG_Train(fareg, device, fareg_train_loader, batch_size=batch2, max_epoch=epoch2,
                          learning_rate=lr2, val_loader=fareg_val_loader, optimizer=fareg_optimizer, verbose = False)

modeldir2 = "models/" + "fareg" + "_ndata" + str(n_train_calib) + "_lr" + str(lr2) +\
            "_delta" + str(delta) + "_nsample" + str(n_sample) + "_ntest" + str(n_test) + "_seed" + str(seed)
fareg_stats = fareg_train.full_train(save_dir = modeldir2)
# print(fareg_stats)

fareg.load_state_dict(torch.load(modeldir2)['model_state'])
fareg.eval()

# predict the worst group
labels = np.array(list(set(y_calib)))
mask_list = []

result_calib_val, _, _ = fareg(torch.Tensor(X_calib[:,-4:]).cuda())
x_calib_val = result_calib_val.cpu().data.numpy()
p_calib_val = np.concatenate([1.0-x_calib_val, x_calib_val], axis=1)

result_test, _, _ = fareg(torch.Tensor(X_test[:,-4:]).cuda())
result_test = result_test.cpu().data
p_test = np.concatenate([1.0-result_test, result_test], axis=1)

for j in range(n_sample):
    sample_calib_val = np.array([np.random.choice([0, 1], size=1, replace=True, p=p_calib_val[i]) for i in range(np.shape(p_calib_val)[0])], dtype=float)
    sample_test = np.array([np.random.choice([0, 1], size=1, replace=True, p=p_test[i]) for i in range(np.shape(p_test)[0])], dtype=float)

    mask_list.append(tuple((list(np.nonzero(np.matrix(sample_calib_val).T[0])[1]), list(np.nonzero(np.matrix(sample_test).T[0])[1]))))



# FaReG
FaReG_method = FaReG_Fairness(alpha = alpha, random_state = seed) 
C_sets_fareg = FaReG_method.multiclass_classification(X_calib, # X_y_calib_val[:,:-1],
                                                            y_calib, # X_y_calib_val[:,-1:].reshape(-1),
                                                            X_test,
                                                            bbox_mc,
                                                            mask_list,  
                                                            conditional = conditional_)

results_marg = eval_psets(C_sets_fareg, y_test)
results_marg = complete_df(results_marg, method = 'FaReG', seed = seed, is_marg_results = True)
results_cond_1 = eval_by_att_mc_1(X_test, y_test, idx_list, C_sets_fareg)
results_cond_1 = complete_df(results_cond_1, method = 'FaReG', seed = seed)
results_cond_2 = eval_by_att_mc_2(X_test, y_test, idx_list, C_sets_fareg)
results_cond_2 = complete_df(results_cond_2, method = 'FaReG', seed = seed)
results_full = pd.concat([results_full, results_marg, results_cond_1, results_cond_2])

wsc_plus = compute_wsc(X_test, y_test, C_sets_fareg, delta=g_delta, M=M)
print("WSC Plus: ", wsc_plus)

################
# Save Results #
################
outfile = outdir + "fareg_" + outfile_name + "_ntest" + str(n_test) + "_seed" + str(seed) + "_delta" + str(delta) + "_nsample" + str(n_sample) + ".csv"
print("Output file: {:s}".format(outfile), end="\n")

results_full.to_csv(outfile, index=False)
print("\nResults written to {:s}\n".format(outfile))
sys.stdout.flush()

# Clean up temp model directory to free up disk space
shutil.rmtree(modeldir + "_seed" + str(seed), ignore_errors=True)
    

    
    
    
