import numpy as np 
import pandas as pd 
import torch
import sys
from scipy.stats.mstats import mquantiles
from statsmodels.stats.multitest import multipletests


def compute_conf_pvals(test_score, calib_scores, left_tail = False):
  n_cal = len(calib_scores)
  if left_tail:
    pval = (1.0 + np.sum(np.array(calib_scores) <= np.array(test_score))) / (1.0 + n_cal)
  else:
    pval = (1.0 + np.sum(np.array(calib_scores) >= np.array(test_score))) / (1.0 + n_cal)
  return pval


def nonconf_scores_mc(X_cal, Y_cal, bbox_mc, alpha = 0.1, random_state = 2023):
  X_cal = torch.from_numpy(X_cal).float()
  Y_cal = torch.from_numpy(Y_cal)
  p_hat_calib = bbox_mc.net.predict_prob(X_cal)
  n_cal = X_cal.shape[0]
  grey_box = ProbAccum(p_hat_calib)
  rng = np.random.default_rng(random_state)
  epsilon = rng.uniform(low=0.0, high=1.0, size=n_cal)
  alpha_max = grey_box.calibrate_scores(Y_cal, epsilon=epsilon)
  scores = alpha - alpha_max
  return scores


class ProbAccum:
    def __init__(self, prob):
        self.n, self.K = prob.shape
        # the label corresponding to sorted probability (from the largest to the smallest)
        self.order = np.argsort(-prob, axis=1)
        # find the rank of each label based on the sorted probability
        self.ranks = np.empty_like(self.order)
        for i in range(self.n):
            self.ranks[i, self.order[i]] = np.arange(len(self.order[i]))
        # sort the predicted probabilities (from the largest to the smallest)
        self.prob_sort = -np.sort(-prob, axis=1)
        # cumsum on the sorted probability
        self.Z = np.round(self.prob_sort.cumsum(axis=1),9)


    def predict_sets(self, alpha, epsilon=None, allow_empty=True):
        if alpha>0:
            # if positive alpha, return L[i] as the largest index that has cumprob greater than 1-alpha
            L = np.argmax(self.Z >= 1.0-alpha, axis=1).flatten()
        else:
            # if alpha = 0, return L[i] be the largest index so that S will be the entire set
            L = (self.Z.shape[1]-1)*np.ones((self.Z.shape[0],)).astype(int)
        if epsilon is not None:
            # Corresponding to the V and U<= V part in equation (5) in the paper
            Z_excess = np.array([ self.Z[i, L[i]] for i in range(self.n) ]) - (1.0-alpha)
            p_remove = Z_excess / np.array([ self.prob_sort[i, L[i]] for i in range(self.n) ])
            remove = epsilon <= p_remove
            for i in np.where(remove)[0]:
                if not allow_empty:
                    # L[i] corresponding to the maximum index (later add probability up to) for the sample i
                    L[i] = np.maximum(0, L[i] - 1)  # Note: avoid returning empty sets
                else:
                    L[i] = L[i] - 1

        # Return prediction set
        S = [ self.order[i,np.arange(0, L[i]+1)] for i in range(self.n) ]
        return(S)

    def calibrate_scores(self, Y, epsilon=None):
        Y = np.atleast_1d(Y)
        if isinstance(Y, int) == False:
          Y = list(map(int, Y))
        n2 = len(Y)
        ranks = np.array([ self.ranks[i,Y[i]] for i in range(n2) ])
        # the cumulative probabilities up to the rank of the true label
        prob_cum = np.array([ self.Z[i,ranks[i]] for i in range(n2) ])
        # the predicted prob for the true label
        prob = np.array([ self.prob_sort[i,ranks[i]] for i in range(n2) ])
        alpha_max = 1.0 - prob_cum
        if epsilon is not None:
            alpha_max += np.multiply(prob, epsilon)
        else:
            alpha_max += prob
        alpha_max = np.minimum(alpha_max, 1)
        return alpha_max

