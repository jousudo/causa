#!/usr/bin/env python3
# Independent numeric oracle for causa.DirectLiNGAM.
#
# This is a self-contained, stdlib-only reimplementation of DirectLiNGAM
# (Shimizu et al., JMLR 2011) that shares NO code with the Go package:
#   - OLS residuals are solved via the normal equations XtX b = Xt y with
#     Gaussian elimination (Go uses Householder QR),
#   - standardization / correlation / residual variance use population (/n)
#     normalization consistently,
#   - the entropy approximation uses the same closed form and constants.
# It emits the fixed dataset as a Go literal plus the quantities the Go test
# locks: pairwise diff-MI, per-candidate exogeneity scores, the recovered
# causal order and the coefficient matrix B.
import math
import random


def make_data():
    # A fixed, strongly non-Gaussian 3-variable SEM so the order is unambiguous.
    # x0 exogenous; x1 = 1.5 x0 + e1; x2 = -0.8 x0 + 0.6 x1 + e2; uniform noise.
    rng = random.Random(20260717)
    n = 14
    x0 = [rng.uniform(-1.0, 1.0) for _ in range(n)]
    x1 = [1.5 * x0[t] + rng.uniform(-1.0, 1.0) for t in range(n)]
    x2 = [-0.8 * x0[t] + 0.6 * x1[t] + rng.uniform(-1.0, 1.0) for t in range(n)]
    return [x0, x1, x2]


def mean(v):
    return sum(v) / len(v)


def standardize(v):
    m = mean(v)
    var = sum((x - m) ** 2 for x in v) / len(v)
    sd = math.sqrt(var)
    return [(x - m) / sd for x in v]


def solve_normal_equations(X, y):
    # X is n rows of k columns; solve least squares by normal equations.
    n = len(X)
    k = len(X[0])
    A = [[0.0] * k for _ in range(k)]
    rhs = [0.0] * k
    for r in range(n):
        for i in range(k):
            rhs[i] += X[r][i] * y[r]
            for j in range(k):
                A[i][j] += X[r][i] * X[r][j]
    # Gaussian elimination with partial pivoting.
    for col in range(k):
        piv = max(range(col, k), key=lambda r: abs(A[r][col]))
        A[col], A[piv] = A[piv], A[col]
        rhs[col], rhs[piv] = rhs[piv], rhs[col]
        for r in range(col + 1, k):
            f = A[r][col] / A[col][col]
            for c in range(col, k):
                A[r][c] -= f * A[col][c]
            rhs[r] -= f * rhs[col]
    coef = [0.0] * k
    for i in range(k - 1, -1, -1):
        s = rhs[i]
        for j in range(i + 1, k):
            s -= A[i][j] * coef[j]
        coef[i] = s / A[i][i]
    return coef


def residual(target, regressor):
    X = [[1.0, regressor[t]] for t in range(len(target))]
    coef = solve_normal_equations(X, target)
    return [target[t] - (coef[0] + coef[1] * regressor[t]) for t in range(len(target))]


def entropy(u):
    k1, k2, gamma = 79.047, 7.4129, 0.37457
    n = len(u)
    s1 = sum(math.log(math.cosh(x)) for x in u) / n
    s2 = sum(x * math.exp(-x * x / 2) for x in u) / n
    return 0.5 * (1 + math.log(2 * math.pi)) - k1 * (s1 - gamma) ** 2 - k2 * s2 ** 2


def diff_mi(xi, xj):
    # xi, xj already standardized.
    rij = standardize(residual(xi, xj))
    rji = standardize(residual(xj, xi))
    return (entropy(xj) + entropy(rij)) - (entropy(xi) + entropy(rji))


def score(std, cand, remaining):
    s = 0.0
    for b in remaining:
        if b == cand:
            continue
        t = diff_mi(std[cand], std[b])
        if t < 0:
            s += t * t
    return s


def most_exogenous(work, remaining):
    std = {v: standardize(work[v]) for v in remaining}
    best, best_s = None, math.inf
    for cand in remaining:
        s = 0.0
        for b in remaining:
            if b == cand:
                continue
            t = diff_mi(std[cand], std[b])
            if t < 0:
                s += t * t
        if s < best_s:
            best_s, best = s, cand
    return best


def direct_lingam(data):
    p = len(data)
    work = [list(v) for v in data]
    remaining = list(range(p))
    order = []
    while len(remaining) > 1:
        m = most_exogenous(work, remaining)
        order.append(m)
        for v in remaining:
            if v != m:
                work[v] = residual(work[v], work[m])
        remaining.remove(m)
    order.append(remaining[0])
    # Coefficients on ORIGINAL data.
    B = [[0.0] * p for _ in range(p)]
    for idx in range(1, len(order)):
        target = order[idx]
        preds = order[:idx]
        X = [[1.0] + [data[pv][t] for pv in preds] for t in range(len(data[0]))]
        coef = solve_normal_equations(X, data[target])
        for c, pv in enumerate(preds):
            B[target][pv] = coef[c + 1]
    return order, B


def go_literal(data):
    rows = []
    for v in data:
        rows.append("\t\t{" + ", ".join(repr(x) for x in v) + "},")
    return "[][]float64{\n" + "\n".join(rows) + "\n\t}"


def main():
    data = make_data()
    print("// --- data literal (paste into the Go test) ---")
    print(go_literal(data))
    print()

    std = [standardize(v) for v in data]
    print("// diff_mi on standardized original columns:")
    print("diffMI(0,1) =", repr(diff_mi(std[0], std[1])))
    print("diffMI(0,2) =", repr(diff_mi(std[0], std[2])))
    print("diffMI(1,2) =", repr(diff_mi(std[1], std[2])))
    print()
    print("// exogeneity scores over {0,1,2} (min-0 squared sum):")
    for cand in range(3):
        print("score(%d) =" % cand, repr(score(std, cand, [0, 1, 2])))
    print()

    order, B = direct_lingam(data)
    print("// recovered order:", order)
    print("// B (b[i][j] = effect of j on i):")
    for i in range(len(B)):
        for j in range(len(B)):
            if B[i][j] != 0.0:
                print("B[%d][%d] =" % (i, j), repr(B[i][j]))


if __name__ == "__main__":
    main()
