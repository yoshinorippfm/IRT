import numpy as np
from functools import partial

# fix to small epsilon
epsilon =  0.0001
# Def of 3 parameter logistic model 
def L3P(a, b, c, x):
    return c + (1 - epsilon - c) / (1 + np.exp(-  a * (x - b)))

# Def of 2 parameter logistic model
def L2P(a, b, c, x):
    return (1 - epsilon) / (1 + np.exp(-  a * (x - b)))

# Def of model parameter
# a > 0 is real , b is real , 0 < c < 1

a_min = 0.7
a_max = 4

b_min = -2
b_max = 2

c_min = 0
c_max = .4

# number of people and question
num_items = 10
num_users = 5_000

# gen of parameter of exam
item_params = np.array(
    [np.random.uniform(a_min, a_max, num_items),
     np.random.uniform(b_min, b_max, num_items),
     np.random.uniform(c_min, c_max, num_items)]
).T

print("初期の問題パラメータの行列")
print(item_params)

# gen of parameter of people
user_params = np.random.normal(size=num_users)

# make a matrix of Item Responce 
# i*j is how question i respond to people j.
ir_matrix_ij = np.vectorize(int)(
    np.array(
        [partial(L3P, *ip)(user_params) > np.random.uniform(0, 1, num_users) for ip in item_params]
    )
)

# remove the out of value.
filter_a = ir_matrix_ij.sum(axis=1) / num_users < 0.9
filter_b = ir_matrix_ij.sum(axis=1) / num_users > 0.1
filter_c = np.corrcoef(np.vstack((ir_matrix_ij, ir_matrix_ij.sum(axis=0))))[num_items][:-1] >= 0.3
filter_total = filter_a & filter_b & filter_c

# Re-Def of the matrix of Item Reponce
irm_ij = ir_matrix_ij[filter_total]
num_items, num_users = irm_ij.shape

X_k = np.linspace(-4, 4, 41)

from scipy.stats import norm
g_k = norm.pdf(X_k)
# E step func
def get_exp_params(irm_ij, g_k, P3_ik):
    Lg_jk = np.exp(irm_ij.T.dot(np.log(P3_ik)) + (1 - irm_ij).T.dot(np.log(1 - P3_ik)))* g_k
    n_Lg_jk = Lg_jk / Lg_jk.sum(axis=1)[:, np.newaxis]
    f_k = n_Lg_jk.sum(axis=0)
    r_ik = irm_ij.dot(n_Lg_jk)
    return f_k, r_ik
# M step's score func
def score_(param, f_k, r_k, X_k):
    a, b, c = param
    P3_k = partial(L3P, a, b, c)(X_k)
    P2_k = partial(L2P, a, b, c)(X_k)
    R_k = r_k / P3_k - f_k
    v = [
        ((X_k - b) * R_k * P2_k).sum(),
        - a * (R_k * P2_k).sum(),
        R_k.sum() / (1 - c)
    ]
    return np.linalg.norm(v)

from scipy.optimize import minimize


cons_ = {
    'type': 'ineq',
    'fun': lambda x:[
        x[0] - a_min,
        a_max - x[0],
        x[1] - b_min,
        b_max - x[1],
        x[2] - c_min,
        c_max - x[2],
    ]
}

a_min, a_max = 0.1, 8.0
b_min, b_max = -4.0, 4.0
c_min, c_max = epsilon, 0.6

# 推定実行用のoarameter

delta = 0.001
# Max of EM algorithm
max_num_of_itr = 1000

# 数値を安定させるために最初に複数回計算して、安定したものの中の中央値を採用する
p_data = []
for n_try in range(10):

    item_params_ = np.array(
        [np.random.uniform(a_min, a_max, num_items),
         np.random.uniform(b_min, b_max, num_items),
         np.random.uniform(c_min, c_max, num_items)]
    ).T

    prev_item_params_ = item_params_
    for itr in range(max_num_of_itr):
        # E step : exp param
        P3_ik = np.array([partial(L3P, *ip)(X_k) for ip in item_params_])
        f_k, r_ik = get_exp_params(irm_ij, g_k, P3_ik)
        ip_n = []
        # resolve of Optimaization problem
        for item_id in range(num_items):
            target = partial(score_, f_k=f_k, r_k=r_ik[item_id], X_k=X_k)
            result = minimize(target, x0=item_params_[item_id], constraints=cons_, method="slsqp")         
            ip_n.append(list(result.x))

        item_params_ = np.array(ip_n)

        mean_diff = abs(prev_item_params_ - item_params_).sum() / item_params_.size
        if mean_diff < delta:
            break
        prev_item_params_ = item_params_

    p_data.append(item_params_)

p_data_ = np.array(p_data)
result_ = []
for idx in range(p_data_.shape[1]):
    t_ = np.array(p_data)[:, idx, :]
  
    filter_1 = t_[:, 1] < b_max - epsilon 
    filter_2 = t_[:, 1] > b_min + epsilon

    result_.append(np.median(t_[filter_1 & filter_2], axis=0))

result = np.array(result_)

print ("result_mtrix columun index is a=識別値, b=困難度, c=当て推量 and row is each probelems")
print (result)