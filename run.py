import os
import sys
from mpmath import mp

mp.prec = 53

ZERO = mp.mpf(0)
APPROX_ZERO = mp.mpf(2**-600)

def log2(x):
    if x == 0:
        return 0
    return mp.log(x, 2)

def p(label, key, val):
    if isinstance(val, int):
        line = "{:s}, {:s}, {:d}".format(label, key, val)
    elif isinstance(val, str):
        line = "{:s}, {:s}, {:s}".format(label, key, val)
    else:
        line = "{:s}, {:s}, {:.1f}".format(label, key, val)
    filename = label.replace("/", "_")
    with open("{}.csv".format(filename), "a") as f:
        print(line, file=f)

def progress(label, output):
    line = "{}: ".format(label) + output
    log_filename = label.replace("/", "_")
    with open("{}.log".format(log_filename), "a") as log:
        print(line, file=log)
    print(line, file=sys.stderr)

def expectation(Pr, f=lambda x: x):
    return sum(f(x)*Pr[x] for x in Pr)

def top_quantile(D, q):
    """ Divides D = { x : Pr[x], ... } into q quantiles.
        Returns conditional distribution of top quantile
    """
    if q == 1:
        return D
    X = D.items()
    X = sorted(X, key=lambda x: x[0])
    i = len(X) - 1
    t = X[i][1]
    while i > 0 and (t + X[i-1][1] <= 1./q):
        i -= 1
        t += X[i][1]
    D2 = dict(X[i:])
    s = 1./sum(D2.values())
    for x in D2:
        D2[x] *= s
    return D2

def tail_probability(D, t):
    '''
    Probability that an drawn from D is strictly greater than t in absolute value
    :param D: Law (Dictionary)
    :param t: tail parameter (integer)
    '''
    s = 0
    for (x,px) in sorted(D.items(), key=lambda t: abs(t[1])):
        if abs(x) > t:
            s += px
    return s

def dist_convolution(A, B, ignore_below=ZERO):
    """ Construct the convolution of two laws (sum of independent variables from two input laws)
    :param A: first input law (dictionnary)
    :param B: second input law (dictionnary)
    """
    C = {}
    for a in A:
        for b in B:
            p = A[a] * B[b]
            if (p > ignore_below):
                C[a+b] = C.get(a+b, ZERO) + p
    return C

def dist_iter_convolution(A, i, ignore_below=APPROX_ZERO):
    """ compute the -ith fold convolution of a distribution (using double-and-add)
    :param A: first input law (dictionnary)
    :param i: (integer)
    """
    D = {0: 1.0}
    i_bin = bin(i)[2:]  # binary representation of n
    for ch in i_bin:
        D = dist_convolution(D, D, ignore_below=ignore_below)
        if ch == '1':
            D = dist_convolution(D, A, ignore_below=ignore_below)
    return D

def dist_scale(A, c):
    """ Assumes A has integer keys and rounds a*c to the first decimal place. """
    B = {}
    for a in A:
        B[round(10 * a * c)/10] = A[a]
    return B

class NTRUHPS:
    def __init__(self, n, q, ephem=False):
        self.n = n
        self.q = q
        self.wt = q//8-2
        self.ephem = ephem

    def threshold(self):
        t = self.q // 2 - 1
        if self.ephem:
            t = t - 1
        return t

    def dfr_average(self):
        t = {-1:1/3, 0:1/3, 1:1/3}
        Dfm = dist_iter_convolution(t, self.wt)
        if self.ephem:
            Dfm = dist_scale(Dfm, 3)
        t = {-3:1/2, 3:1/2}
        Dgr = dist_iter_convolution(t, self.wt)
        return dist_convolution(Dfm, Dgr)

    def dfr_top_quantile(self, lgu):
        """ Failure probability when f is in top 2^lgu quantile in terms of
            length (which occurs with probability 2^-lgu).

            Assumes adversary chooses r with weight n-1.
        """
        t = {0: 1/3, 1:2/3}
        D = dist_iter_convolution(t, self.n-1)
        pnz = expectation(top_quantile(D, 2**(lgu//2)))/(self.n-1)

        t = {-1:pnz/2, 0:1-pnz, 1:pnz/2}
        Dfm = dist_iter_convolution(t, self.wt)
        if self.ephem:
            Dfm = dist_scale(Dfm, 3)
        t = {-3:1/2, 3:1/2}
        Dgr = dist_iter_convolution(t, self.wt)
        return dist_convolution(Dfm, Dgr)


class NTRUHRSS:
    def __init__(self, n, q=None, ephem=False):
        self.n = n
        self.q = q if q else int(2**round(0.5 + 3.5 + log2(n)))
        self.ephem = ephem

    def threshold(self):
        t = (self.q // 2 - 1)/(2**0.5)
        if self.ephem:
            t = t-1
        return t

    def dfr_average(self):
        t = {-1:1/3, 0:1/3, 1:1/3}
        Dfm = dist_iter_convolution(t, self.n-1)
        Dgr = dist_scale(Dfm, 3)
        if self.ephem:
            Dfm = dist_scale(Dfm, 3)
        return dist_convolution(Dfm, Dgr)

    def dfr_top_quantile(self, lgu):
        """ Failure probability when f is in top 2^lgu quantile in terms of
            length (which occurs with probability 2^-lgu).

            Assumes adversary chooses r with weight n-1.
        """
        t = {0: 1/3, 1:2/3}
        D = dist_iter_convolution(t, self.n-1)
        pnz = expectation(top_quantile(D, 2**(lgu/2)))/(self.n-1)
        t = {-1:pnz/2, 0:1-pnz, 1:pnz/2}
        Dfm = dist_iter_convolution(t, self.n-1)
        Dgr = dist_scale(Dfm, 3)
        if self.ephem:
            Dfm = dist_scale(Dfm, 3)
        return dist_convolution(Dfm, Dgr)

if __name__ == "__main__":
    from multiprocessing import Pool, cpu_count
    NCORES = cpu_count()

    PSS = [
           ("ephem509", NTRUHPS(509,  2048, ephem=True)),
           ("ephem677", NTRUHPS(677,  2048, ephem=True)),
           ("ephem701", NTRUHRSS(701, 8192, ephem=True)),
           ("ephem821", NTRUHPS(821,  4096, ephem=True))
          ]

    def do(label, ps, lgsq):
        progress(label, "starting f/{}".format(lgsq))
        dfr = tail_probability(ps.dfr_top_quantile(lgsq), ps.threshold())
        p(label, "{}".format(lgsq), float(log2(ps.n*dfr)))
        progress(label, "done f/{}".format(lgsq))

    def __do(args):
        do(*args)

    jobs = []

    FQS = [0, 64]
    for (label, ps) in PSS:
        for i in FQS:
            jobs.append((label, ps, i))

    list(Pool(NCORES).imap_unordered(__do, jobs))

