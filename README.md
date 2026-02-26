# FaReG (Fair Conformal Prediction for Representation-Based Groups)
Conformal prediction methods provide statistically rigorous marginal coverage guarantees for machine learning models, but such guarantees fail to account for algorithmic biases, thereby undermining fairness and trust. This paper introduces a fair conformal inference framework for classification tasks. The proposed method constructs prediction sets that guarantee conditional coverage on adaptively identified subgroups, which can be implicitly defined through nonlinear feature combinations. By balancing effectiveness and efficiency in producing compact, informative prediction sets and ensuring adaptive equalized coverage across unfairly treated subgroups, our approach paves a practical pathway toward trustworthy machine learning. Extensive experiments on both synthetic and real-world datasets demonstrate the effectiveness of the framework.

Accompanying paper: *Fair Conformal Classification via Learning Representation-Based Groups*.


## Contents

 - `Group_fairness` Python package implementing our method.
    - `Group_fairness/blackbox.py` Codes to train and evaluate models.
    - `Group_fairness/evals.py` Codes to evaluate the performance of conformal prediction sets.
    - `Group_fairness/cp.py` Codes to build conformal prediction sets. 
    - `Group_fairness/networks.py` Deep networks.
    - `Group_fairness/wsc.py` WSC metrics.
 - `third_party/` Third-party Python packages.
 - `experiment/` Codes to replicate the experiments in the accompanying paper.
    - `experiment/method.py` Codes to reproduce the main results for multi-class classification tasks.
    - `experiment/data_gen.py` Codes to generate the dataset used in the accompanying paper.  


    
## Prerequisites

Prerequisites for the FaReG package:
 - numpy
 - scipy
 - sklearn
 - torch
 - random
 - pathlib
 - tqdm
 - math
 - pandas
 - matplotlib
 - torchmetrics
 - statsmodels

Additional prerequisites to run the experiments:
 - shutil
 - tempfile
 - pickle
 - sys
 - os
