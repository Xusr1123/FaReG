import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests


def eval_psets(S, y):
  coverage = np.mean([y[i] in S[i] for i in range(len(y))])
  length = np.mean([len(S[i]) for i in range(len(y))])
  idx_cover = np.where([y[i] in S[i] for i in range(len(y))])[0]
  length_cover = np.mean([len(S[i]) for i in idx_cover])
  result = pd.DataFrame({'Coverage': [coverage], 'Length': [length], 'Length_cover': [length_cover]})
  return result


def eval_by_att_mc_1(X_test, y_test, att_idx_full, C_sets):
  '''Input a np.array as test features,
  alpha is a list'''
  results = pd.DataFrame()
  for att_idx in att_idx_full:
    for level in set(X_test[:,att_idx]):
      # print('att_idx, level: ', att_idx, level)
      group_ = np.where(X_test[:,att_idx]==level)
      result_ = eval_psets(np.array(C_sets, dtype = 'object')[group_], y_test[group_])
      result_['n_count'] = sum(X_test[:,att_idx] == level)
      # result_['attribute_level'] = str(att_idx) + str('_') + str(level)
      result_['attribute_idx'] = att_idx
      result_['attribute_level'] = level
      results = pd.concat([results, result_])

  return results


def eval_by_att_mc_2(X_test, y_test, att_idx_full, C_sets):
  '''Input a np.array as test features,
  alpha is a list'''
  results = pd.DataFrame()
  for att_idx1 in att_idx_full:
    for att_idx2 in att_idx_full:
      if att_idx1 >= att_idx2:
         continue

      for level1 in set(X_test[:,att_idx1]):
        for level2 in set(X_test[:,att_idx2]):
          # print('att_idx1, level1: ', att_idx1, level1)
          # print('att_idx2, level2: ', att_idx2, level2)

          group_ = np.where((X_test[:,att_idx1]==level1) & (X_test[:,att_idx2]==level2))
          result_ = eval_psets(np.array(C_sets, dtype = 'object')[group_], y_test[group_])
          result_['n_count'] = sum((X_test[:,att_idx1]==level1) & (X_test[:,att_idx2]==level2))
          # result_['attribute_level'] = str(att_idx) + str('_') + str(level)
          result_['attribute_idx'] = str(att_idx1) + '+' + str(att_idx2)
          result_['attribute_level'] = str(level1) + '+' + str(level2)
          results = pd.concat([results, result_])

  return results

