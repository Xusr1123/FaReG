import numpy as np
import tqdm
from sklearn.model_selection import train_test_split
from multiprocessing import Pool, cpu_count

def wsc_v_1(X, y, S, delta, v):
    n = len(y)
    cover = np.array([y[i] in S[i] for i in range(n)])
    z = np.dot(X,v)
    # Compute mass
    z_order = np.argsort(z)
    z_sorted = z[z_order]
    cover_ordered = cover[z_order]
    ai_max = int(np.round((1.0 - delta) * n))
    ai_best = 0
    bi_best = n - 1
    cover_min = 1
    for ai in np.arange(0, ai_max):
        bi_min = np.minimum(ai + int(np.round(delta * n)), n)
        coverage = np.cumsum(cover_ordered[ai:n]) / np.arange(1, n - ai + 1)
        coverage[np.arange(0, bi_min - ai)] = 1
        bi_star = ai + np.argmin(coverage)
        cover_star = coverage[bi_star - ai]
        if cover_star < cover_min:
            ai_best = ai
            bi_best = bi_star
            cover_min = cover_star
    return cover_min, z_sorted[ai_best], z_sorted[bi_best], [ai_best, bi_best, z_order]


def wsc_v_2(X, y, S, delta, V, w):
    n = len(y)
    cover = np.array([y[i] in S[i] for i in range(n)])
    z_matrix = np.dot(np.dot(X, V), X.T)
    z_1 = z_matrix.diagonal()
    z_2 = np.dot(X, w)
    z = z_1 + z_2
    # Compute mass
    z_order = np.argsort(z)
    z_sorted = z[z_order]
    cover_ordered = cover[z_order]
    ai_max = int(np.round((1.0 - delta) * n))
    ai_best = 0
    bi_best = n - 1
    cover_min = 1
    for ai in np.arange(0, ai_max):
        bi_min = np.minimum(ai + int(np.round(delta * n)), n)
        coverage = np.cumsum(cover_ordered[ai:n]) / np.arange(1, n - ai + 1)
        coverage[np.arange(0, bi_min - ai)] = 1
        bi_star = ai + np.argmin(coverage)
        cover_star = coverage[bi_star - ai]
        if cover_star < cover_min:
            ai_best = ai
            bi_best = bi_star
            cover_min = cover_star
    return cover_min, z_sorted[ai_best], z_sorted[bi_best], [ai_best, bi_best, z_order]


def mp_wsc(X, y, S, delta=0.3, M=1000, random_state=2025, verbose=False):
    cpu_count = 15
    rng = np.random.default_rng(random_state)

    def sample_sphere(n, p):
        v = rng.normal(size=(p, n))
        v /= np.linalg.norm(v, axis=0)
        return v.T

    V = sample_sphere(M, p=X.shape[1])
    
    params = [(X, y, S, delta, V[m]) for m in range(M)]
    
    with Pool(processes = cpu_count) as pool:
        results = pool.starmap(wsc_v_1, params)
    
    wsc_list, a_list, b_list, c_list = zip(*results)
    wsc_list = list(wsc_list)
    a_list = list(a_list)
    b_list = list(b_list)
    c_list = list(c_list)
    
    # idx_star = np.argmin(np.array(wsc_list))
    # a_star = a_list[idx_star]
    # b_star = b_list[idx_star]
    # c_star = c_list[idx_star]
    # v_star = V[idx_star]
    # wsc_star = wsc_list[idx_star]
    # return wsc_star, v_star, a_star, b_star, c_star
    
    return wsc_list



def mp_wsc_plus(X, y, S, delta=0.3, M=1000, random_state=2025, verbose=False):
    cpu_count = 10
    rng = np.random.default_rng(random_state)

    def sample_sphere_v(n):
        v = rng.normal(size=(n, n))
        v /= np.linalg.norm(v, axis=0)
        return v
    
    def sample_sphere_w(n, p):
        w = rng.normal(size=(p, n))
        w /= np.linalg.norm(w, axis=0)
        return w.T

    V_list = []
    for _ in range(M):
        V = sample_sphere_v(X.shape[1])
        V_list.append(V)
        
    W = sample_sphere_w(M, p=X.shape[1])
    
    params = [(X, y, S, delta, V_list[m], W[m]) for m in range(M)]
    
    with Pool(processes = cpu_count) as pool:
        results = pool.starmap(wsc_v_2, params)
    
    wsc_list, a_list, b_list, c_list = zip(*results)
    wsc_list = list(wsc_list)
    a_list = list(a_list)
    b_list = list(b_list)
    c_list = list(c_list)
        
    # idx_star = np.argmin(np.array(wsc_list))
    # a_star = a_list[idx_star]
    # b_star = b_list[idx_star]
    # c_star = c_list[idx_star]
    # v_star = V_list[idx_star]
    # wsc_star = wsc_list[idx_star]
    # return wsc_star, v_star, a_star, b_star, c_star

    return wsc_list



def wsc(X, y, S, delta=0.3, M=1000, random_state=2025, verbose=False):
    rng = np.random.default_rng(random_state)

    def wsc_v(X, y, S, delta, v):
        n = len(y)
        cover = np.array([y[i] in S[i] for i in range(n)])
        z = np.dot(X,v)
        # Compute mass
        z_order = np.argsort(z)
        z_sorted = z[z_order]
        cover_ordered = cover[z_order]
        ai_max = int(np.round((1.0 - delta) * n))
        ai_best = 0
        bi_best = n - 1
        cover_min = 1
        for ai in np.arange(0, ai_max):
            bi_min = np.minimum(ai + int(np.round(delta * n)), n)
            coverage = np.cumsum(cover_ordered[ai:n]) / np.arange(1, n - ai + 1)
            coverage[np.arange(0, bi_min - ai)] = 1
            bi_star = ai + np.argmin(coverage)
            cover_star = coverage[bi_star - ai]
            if cover_star < cover_min:
                ai_best = ai
                bi_best = bi_star
                cover_min = cover_star
        return cover_min, z_sorted[ai_best], z_sorted[bi_best], [ai_best, bi_best, z_order]

    def sample_sphere(n, p):
        v = rng.normal(size=(p, n))
        v /= np.linalg.norm(v, axis=0)
        return v.T

    V = sample_sphere(M, p=X.shape[1])
    wsc_list = [[]] * M
    a_list = [[]] * M
    b_list = [[]] * M
    c_list = [[]] * M

    for m in range(M):
        wsc_list[m], a_list[m], b_list[m], c_list[m] = wsc_v(X, y, S, delta, V[m])
        
    # idx_star = np.argmin(np.array(wsc_list))
    # a_star = a_list[idx_star]
    # b_star = b_list[idx_star]
    # c_star = c_list[idx_star]
    # v_star = V[idx_star]
    # wsc_star = wsc_list[idx_star]
    # return wsc_star, v_star, a_star, b_star, c_star
    
    return wsc_list



def wsc_plus(X, y, S, delta=0.3, M=1000, random_state=2025, verbose=False):
    rng = np.random.default_rng(random_state)

    def wsc_v(X, y, S, delta, V, w):
        n = len(y)
        cover = np.array([y[i] in S[i] for i in range(n)])
        z_matrix = np.dot(np.dot(X, V), X.T)
        z_1 = z_matrix.diagonal()
        z_2 = np.dot(X, w)
        z = z_1 + z_2
        # Compute mass
        z_order = np.argsort(z)
        z_sorted = z[z_order]
        cover_ordered = cover[z_order]
        ai_max = int(np.round((1.0 - delta) * n))
        ai_best = 0
        bi_best = n - 1
        cover_min = 1
        for ai in np.arange(0, ai_max):
            bi_min = np.minimum(ai + int(np.round(delta * n)), n)
            coverage = np.cumsum(cover_ordered[ai:n]) / np.arange(1, n - ai + 1)
            coverage[np.arange(0, bi_min - ai)] = 1
            bi_star = ai + np.argmin(coverage)
            cover_star = coverage[bi_star - ai]
            if cover_star < cover_min:
                ai_best = ai
                bi_best = bi_star
                cover_min = cover_star
        return cover_min, z_sorted[ai_best], z_sorted[bi_best], [ai_best, bi_best, z_order]

    def sample_sphere_v(n):
        v = rng.normal(size=(n, n))
        v /= np.linalg.norm(v, axis=0)
        return v
    
    def sample_sphere_w(n, p):
        w = rng.normal(size=(p, n))
        w /= np.linalg.norm(w, axis=0)
        return w.T

    V_list = []
    for _ in range(M):
        V = sample_sphere_v(X.shape[1])
        V_list.append(V)
        
    W = sample_sphere_w(M, p=X.shape[1])
    
    wsc_list = [[]] * M
    a_list = [[]] * M
    b_list = [[]] * M
    c_list = [[]] * M

    for m in range(M):
        wsc_list[m], a_list[m], b_list[m], c_list[m] = wsc_v(X, y, S, delta, V_list[m], W[m])
        
    # idx_star = np.argmin(np.array(wsc_list))
    # a_star = a_list[idx_star]
    # b_star = b_list[idx_star]
    # c_star = c_list[idx_star]
    # v_star = V_list[idx_star]
    # wsc_star = wsc_list[idx_star]
    # return wsc_star, v_star, a_star, b_star, c_star

    return wsc_list







