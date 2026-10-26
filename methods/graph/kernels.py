"""Скомпилированные (numba) циклы для горячих мест графовых методов.

Каждое ядро делает ровно то же, что numpy-версия рядом с ним (её держим как эталон для тестов):
тот же порядок батчей и те же случайные числа, отличие — только в порядке сложения (ошибки ~1e-15).
Компиляция при первом вызове (несколько секунд), дальше — из кэша в __pycache__.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit, prange


@njit(cache=True, inline="always")
def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


@njit(cache=True, parallel=True)
def sgns_neg_epoch(W_in, W_out, c, x, negs, lrs, batch):
    """Мини-батчи negative sampling: градиенты батча считаются по весам до шага (параллельно по парам),
    затем применяются по порядку — результат не зависит от числа потоков."""
    n, d = len(c), W_in.shape[1]
    K = negs.shape[1]
    g_in = np.empty((batch, d))
    g_out = np.empty((batch, K + 1, d))
    lp = np.zeros(batch)
    loss = 0.0
    for bi in range(len(lrs)):
        s, e = bi * batch, min(n, (bi + 1) * batch)
        for i in prange(s, e):
            v = W_in[c[i]]
            up = W_out[x[i]]
            sp = 0.0
            for t in range(d):
                sp += v[t] * up[t]
            li = -math.log(_sigmoid(sp) + 1e-300)
            dp = _sigmoid(sp) - 1.0
            for t in range(d):
                g_in[i - s, t] = dp * up[t]
                g_out[i - s, 0, t] = dp * v[t]
            for k in range(K):
                un = W_out[negs[i, k]]
                sn = 0.0
                for t in range(d):
                    sn += v[t] * un[t]
                li -= math.log(_sigmoid(-sn) + 1e-300)
                dn = _sigmoid(sn)
                for t in range(d):
                    g_in[i - s, t] += dn * un[t]
                    g_out[i - s, k + 1, t] = dn * v[t]
            lp[i - s] = li
        for i in range(s, e):
            loss += lp[i - s]
        lr = lrs[bi]
        for i in range(s, e):
            for t in range(d):
                W_in[c[i], t] -= lr * g_in[i - s, t]
        for i in range(s, e):
            for t in range(d):
                W_out[x[i], t] -= lr * g_out[i - s, 0, t]
            for k in range(K):
                for t in range(d):
                    W_out[negs[i, k], t] -= lr * g_out[i - s, k + 1, t]
    return loss


@njit(cache=True, parallel=True)
def sgns_hs_epoch(W_in, W_out, c, x, paths, codes, lens, lrs, batch):
    """Мини-батчи hierarchical softmax по дереву Хаффмана, та же схема шага."""
    n, d = len(c), W_in.shape[1]
    Lmax = paths.shape[1]
    g_in = np.empty((batch, d))
    g_out = np.empty((batch, Lmax, d))
    lp = np.zeros(batch)
    loss = 0.0
    for bi in range(len(lrs)):
        s, e = bi * batch, min(n, (bi + 1) * batch)
        for i in prange(s, e):
            v = W_in[c[i]]
            ctx = x[i]
            for t in range(d):
                g_in[i - s, t] = 0.0
            li = 0.0
            for j in range(lens[ctx]):
                th = W_out[paths[ctx, j]]
                sign = 1.0 - 2.0 * codes[ctx, j]
                z = 0.0
                for t in range(d):
                    z += v[t] * th[t]
                z *= sign
                li -= math.log(_sigmoid(z) + 1e-300)
                ds = (_sigmoid(z) - 1.0) * sign
                for t in range(d):
                    g_in[i - s, t] += ds * th[t]
                    g_out[i - s, j, t] = ds * v[t]
            lp[i - s] = li
        for i in range(s, e):
            loss += lp[i - s]
        lr = lrs[bi]
        for i in range(s, e):
            for t in range(d):
                W_in[c[i], t] -= lr * g_in[i - s, t]
        for i in range(s, e):
            ctx = x[i]
            for j in range(lens[ctx]):
                for t in range(d):
                    W_out[paths[ctx, j], t] -= lr * g_out[i - s, j, t]
    return loss


@njit(cache=True)
def dtw(a, ca, b, cb, empty):
    """DTW сжатых последовательностей степеней, цена (max/min - 1) * max(count) — как struc2vec.dtw."""
    if len(a) == 0 and len(b) == 0:
        return 0.0
    if len(a) == 0:
        a, ca = np.array([empty]), np.array([1.0])
    if len(b) == 0:
        b, cb = np.array([empty]), np.array([1.0])
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            hi, lo = max(a[i - 1], b[j - 1]), min(a[i - 1], b[j - 1])
            cost = (hi / lo - 1.0) * max(ca[i - 1], cb[j - 1])
            D[i, j] = cost + min(D[i - 1, j], D[i - 1, j - 1], D[i, j - 1])
    return D[n, m]
