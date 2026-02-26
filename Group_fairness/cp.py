import numpy as np 
from statsmodels.stats.multitest import multipletests 
from scipy import stats 
import sys 
from tqdm import tqdm

sys.path.append("../")
from third_party.utils import compute_conf_pvals, nonconf_scores_mc


class Marginal_Fairness:
    def __init__(self, alpha, random_state = 2025):
        self.alpha = alpha
        self.random_state = random_state   

    def multiclass_classification(self, X_tests, X_calib, Y_calib, bbox_mc, left_tail = False, conditional = True):
        cal_scores = nonconf_scores_mc(X_calib, Y_calib, bbox_mc, alpha = self.alpha, random_state = self.random_state)
        C_full = []
        n_test = X_tests.shape[0]
        conf_pval = np.full((n_test, len(set(Y_calib))), -np.Inf)
        
        for idx, y in enumerate(set(Y_calib)):
            # print('idx, y: ', idx, y)
            scores_test_y = nonconf_scores_mc(X_tests,
                                              np.repeat(y, n_test),
                                              bbox_mc,
                                              alpha = self.alpha,
                                              random_state = self.random_state)

            idx_y = [np.where(Y_calib == y) if conditional else np.arange(len(Y_calib))][0]
            cal_scores_y = cal_scores[idx_y]
            
            for i, X_test in enumerate(X_tests):
                # print('i, X_test: ', i, X_test)
                test_score = scores_test_y[i]
                # print('test_score: ', test_score)
                conf_pval[i, idx] = compute_conf_pvals(test_score, cal_scores_y, left_tail = left_tail)
                # print(compute_conf_pvals(test_score, cal_scores_y, left_tail = left_tail))

                
        for i in np.arange(n_test):
            C_set = np.where(np.array(conf_pval[i]) >= self.alpha)[0].tolist()
            C_full.append(C_set)
            
        return C_full
            
            
            
class FaReG_Fairness:

    def __init__(self, alpha, random_state = 2025):
        self.alpha = alpha
        self.random_state = random_state

    def multiclass_classification(self, X_calib, Y_calib, X_test, bbox_mc, mask_list, conditional = False, left_tail = False):

        n_test = X_test.shape[0]
        labels = np.array(list(set(Y_calib)))

        k_final = []
        C_sets_final = []

        calib_scores = nonconf_scores_mc(X_calib, Y_calib, bbox_mc, alpha = self.alpha, random_state = self.random_state)
        # print('calib_scores: ', calib_scores)
        test_scores = np.full((n_test, len(labels)), -np.inf)
        conf_pval_y = np.full((n_test, len(labels)), -np.Inf)
        conf_pval_add = np.full((n_test, len(labels)), -np.Inf)

        for idx, y in enumerate(labels):
            test_scores[:, idx] = nonconf_scores_mc(X_test, np.repeat(y,n_test), bbox_mc, alpha = self.alpha, random_state = self.random_state)
            # print('test_scores: ', test_scores[:, idx])

        for i, X in tqdm(enumerate(X_test)):
            for idx, y in enumerate(labels):
                idx_y = [np.where(Y_calib == y) if conditional else np.arange(len(Y_calib))][0]

                calib_scores_y = calib_scores[idx_y]
                conf_pval_add[i, idx] = compute_conf_pvals(test_scores[i, idx], calib_scores_y, left_tail = left_tail)
                
                for mask in mask_list:
                    if i not in mask[1]:
                        calib_scores_selected = calib_scores_y
                        # print('here1: ', len(calib_scores_selected))
                    else:
                        calib_scores_selected = np.array(calib_scores_y)[mask[0]]

 
                    conf_pval_y[i, idx] = max(compute_conf_pvals(test_scores[i, idx], calib_scores_selected, left_tail = left_tail), conf_pval_y[i, idx])
                
          
            # The prediction sets constructed using the place-holder label y
            C_set_y = set(labels[np.where(np.array(conf_pval_y[i]) >= self.alpha)[0]])
            C_set_add = set(labels[np.where(np.array(conf_pval_add[i]) >= self.alpha)[0]])
            print(C_set_y, C_set_add)
            print('differences: ', C_set_y.difference(C_set_add))
            # Take the union
            C_set_ = C_set_y.union(C_set_add)
            C_sets_final.append(list(C_set_))
            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

        return C_sets_final


