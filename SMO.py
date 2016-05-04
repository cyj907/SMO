import random


class SMO:

    def __init__(self, X, Y, C):
        self.X = X                      # data matrix, size: m x n, m: number of samples, n: number of features
        self.Y = Y                      # target labels, size: n
        self.C = C                      # parameter
        self.tol = 0.01                 # KKT violation tolerance
        self.m = len(X)                 # sample size
        self.n = len(X[0])              # feature size
        self.alpha = [0 for i in range(self.m)]         # lagrange multipliers
        self.b = 0                                      # threshold b
        self.error_cache = [0 for i in range(self.m)]   # store error

        # initialize error cache
        for i in range(self.m):
            self.error_cache[i] = self.svm_func(i) - self.Y[i]

    # main procedure
    def apply(self):
        num_changed = 0
        examine_all = 1
        while num_changed > 0 or examine_all == 1:
            num_changed = 0
            if examine_all:
                for i in range(self.m):
                    num_changed += self.examineExample(i)
            else:
                for i in range(self.m):
                    if 0 < self.alpha[i] < self.C:
                        num_changed += self.examineExample(i)
            if examine_all:
                examine_all = 0
            elif num_changed == 0:
                examine_all = 1

    def examineExample(self, i2):
        y2 = self.Y[i2]
        alpha2 = self.alpha[i2]
        E2 = self.error_cache[i2]
        r2 = E2 * y2

        # violate KKT
        if r2 < - self.tol and alpha2 < self.C or r2 > self.tol and alpha2 > 0:
            if self.count_non_bounded() > 1:
                i1 = self.get_second_choice(i2)
                if self.takeStep(i1, i2):
                    return True

            # loop over all non-zero and non-C alpha, starting at random point
            st = random.randint(0, self.m - 1)
            en = st
            while st < self.m:
                if self.alpha[st] != 0 and self.alpha[st] != self.C:
                    if self.takeStep(st, i2):
                        return True
                st += 1
            st = 0
            while st < en:
                if self.alpha[st] != 0 and self.alpha[st] != self.C:
                    if self.takeStep(st, i2):
                        return True
                st += 1

            # loop over all possible i1, starting at a random point
            st = random.randint(0, self.m - 1)
            en = st
            while st < self.m:
                if self.takeStep(st, i2):
                    return True
                st += 1
            st = 0
            while st < en:
                if self.takeStep(st, i2):
                    return True
                st += 1

        return False

    def takeStep(self, i1, i2):
        eps = 1e-8

        if i1 == i2:
            return False

        alpha1 = self.alpha[i1]
        y1 = self.Y[i1]
        E1 = self.error_cache[i1]

        alpha2 = self.alpha[i2]
        y2 = self.Y[i2]
        E2 = self.error_cache[i2]

        # compute L, H
        s = y1 * y2
        if s == 1:
            L = max(0, alpha1 + alpha2 - self.C)
            H = min(self.C, alpha1 + alpha2)
        else:
            L = max(0, alpha2 - alpha1)
            H = min(self.C, self.C + alpha2 - alpha1)

        if L == H:
            return False

        k11 = self.kernel(i1, i1)
        k12 = self.kernel(i1, i2)
        k22 = self.kernel(i2, i2)

        eta = 2 * k12 - k11 - k22
        if eta < 0:
            # the objective function is concave (bowl down)
            a2 = alpha2 - y2 * (E1 - E2) / eta

            # clip alpha2
            if a2 < L:
                a2 = L
            elif a2 > H:
                a2 = H
        else:
            # the objective function is not concave (bowl up)
            # the maximum of objective function stays at bound (L or H)
            tmp = self.alpha[i2]
            self.alpha[i2] = L
            Lobj = self.obj_func()
            self.alpha[i2] = H
            Hobj = self.obj_func()
            self.alpha[i2] = tmp

            if Lobj > Hobj + eps:
                a2 = L
            elif Lobj < Hobj - eps:
                a2 = H
            else:
                # the objective function is the same at both ends
                # joint maximization cannot make progress
                # give up updating a2
                a2 = alpha2

        if a2 < eps:
            a2 = 0
        elif a2 > self.C - eps:
            a2 = self.C

        if abs(a2 - alpha2) < eps * (a2 + alpha2 + eps):
            return False

        # update a1
        a1 = alpha1 + s * (alpha2 - a2)

        # Update threshold to reflect change in Lagrange multipliers
        old_b = self.b
        self.update_b(i1, i2, a1, a2)

        # Update error cache using new Lagrange multipliers
        self.update_error_cache(i1, i2, a1, a2, old_b)

        # Store a1 and a2 in the alpha array
        self.alpha[i1] = a1
        self.alpha[i2] = a2

        return True

    # decision function for svm
    def svm_func(self, i1):
        s = 0
        for i in range(self.m):
            s += self.alpha[i] * self.Y[i] * self.kernel(i, i1)
        return s - self.b

    # objective function for dual problem
    def obj_func(self):
        s = 0
        for i in range(self.m):
            s += self.alpha[i]

        s2 = 0
        for i in range(self.m):
            for j in range(self.m):
                s2 += self.Y[i] * self.Y[j] * self.kernel(i, j) * self.alpha[i] * self.alpha[j]

        return s - s2 / 2

    # kernel, linear in this case
    def kernel(self, i, j):
        x1 = self.X[i]
        x2 = self.X[j]

        s = 0
        for i in range(self.n):
            s += x1[i] * x2[i]
        return s

    # update threshold b
    def update_b(self, i1, i2, a1, a2):
        b1 = self.error_cache[i1] + self.Y[i1] * (a1 - self.alpha[i1]) * self.kernel(i1,i1) \
            + self.Y[i2] * (a2 - self.alpha[i2]) * self.kernel(i1, i2) + self.b
        b2 = self.error_cache[i2] + self.Y[i1] * (a1 - self.alpha[i1]) * self.kernel(i1,i2) \
            + self.Y[i2] * (a2 - self.alpha[i2]) * self.kernel(i2, i2) + self.b

        is_b1_valid = False
        is_b2_valid = False
        if 0 < a1 < self.C:
            is_b1_valid = True
        if 0 < a2 < self.C:
            is_b2_valid = True

        if is_b1_valid:
            self.b = b1
        elif is_b2_valid:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2

    # update error cache
    def update_error_cache(self, i1, i2, a1, a2, old_b):
        for i in range(self.m):
            self.error_cache[i] = self.error_cache[i] + self.Y[i1] * (a1 - self.alpha[i1]) * self.kernel(i1, i)\
                        + self.Y[i2] * (a2 - self.alpha[i2]) * self.kernel(i2, i) + old_b - self.b

    # get the number of non-bounded samples
    def count_non_bounded(self):
        cnt = 0
        for i in range(self.m):
            if 0 < self.alpha[i] < self.C:
                cnt += 1
        return cnt

    # compute second choice based on maximizing |E2 - E1|
    def get_second_choice(self, i2):
        i1 = 0
        if self.error_cache[i2] >= 0:
            # pick up the i with min error
            for i in range(1, self.m):
                if self.error_cache[i] < self.error_cache[i1]:
                    i1 = i
        else:
            # pick up the i with max error
            for i in range(1, self.m):
                if self.error_cache[i] > self.error_cache[i1]:
                    i1 = i

        return i1

    # get w and b for linear kernel svm
    def get_params(self):
        w = [0 for i in range(self.n)]
        for i in range(self.m):
            x = self.X[i]

            for j in range(self.n):
                w[j] += self.alpha[i] * self.Y[i] * x[j]

        return w, self.b



