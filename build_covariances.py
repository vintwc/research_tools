import numpy as np
import math
from sklearn.isotonic import IsotonicRegression as IR
import pandas as pd


def winsorization(r):
    import scipy.stats as st
    # Calculate the trimmed mean for each column
    r_means = r.apply(lambda x: st.trim_mean(x, 0.1), axis=0)

    # Calculate the MAD for each column
    r_mads = r.apply(lambda x: st.median_abs_deviation(x), axis=0)

    # Determine the lower and upper bounds for winsorization
    r_lows = r_means - 5 * r_mads
    r_highs = r_means + 5 * r_mads

    # Apply winsorization by clipping the DataFrame's values to the bounds
    r_ws = r.clip(lower=r_lows, upper=r_highs, axis=1)

    return r_ws


class CovarianceEstimator:
    def __init__(self, input_dict):
        self.X = None
        self.N = None
        self.T = None
        self.estimator = None
        self.beta = input_dict['beta']
        self.n_fold = input_dict['n_fold']
        self.n_fac = input_dict['n_fac']
        self.devol = input_dict['devol']

    @staticmethod
    def corr2cov(corr, std):
        cov = corr
        cov *= std
        cov *= std.reshape(-1, 1)
        return cov

    @staticmethod
    def igarch(ret):
        # compute initial variance
        var_init = ret.var().to_frame().transpose()

        # join the initial variance with the remaining data
        data = pd.concat([var_init, ret[:-1] ** 2], ignore_index=True)
        data.index = ret.index

        # compute volatility using Pandas ewm
        univ_vol = np.sqrt(data.ewm(alpha=(1 - 0.97), adjust=False).mean())
        return univ_vol

    def time_series_norm(self, ret):
        # compute vol
        vol = self.igarch(ret)

        # vol adjust
        ret_adjust = (ret / vol).clip(-4.2, 4.2)
        return ret_adjust, vol.iloc[-1].values

    def wrapper_cov_est(self, X):
        self.T, self.N = X.shape
        # X = winsorization(X)
        if self.devol:
            X, self.vol = self.time_series_norm(X)
        else:
            self.vol = np.ones(self.N)

            # Define a dictionary mapping estimator names to their corresponding methods
            estimator_methods = {
                'ewa_cv': lambda X: self.get_CV(X, beta=self.beta, n_fold=self.n_fold),
                'cv': lambda X: self.get_CV(X, n_fold=self.n_fold),
                'nl': lambda X: self.get_QIS(X),
                'ls': lambda X: self.get_LS(X),
                'id': lambda X: np.eye(self.N)
            }

            # Use the dictionary to select and call the appropriate method
            if self.estimator in estimator_methods:
                Sigma_tmp = estimator_methods[self.estimator](X)
            else:
                raise NotImplementedError(f"Covariance estimator {self.estimator} not implemented")

            Sigma = self.corr2cov(Sigma_tmp, self.vol)
            return Sigma

    def low_rank_approx(self, cov):
        d, v = np.linalg.eigh(cov)
        eigenvecs = v[:, -self.n_fac:]
        eigenvals_diag = np.diag(d[-self.n_fac:])
        var_noise = (np.trace(cov) - np.trace(eigenvecs @ eigenvals_diag @ eigenvecs.T)) / self.N
        cov_noise = var_noise * np.eye(self.N)
        return eigenvecs, eigenvals_diag, cov_noise


    def get_LS(self, Y, k=None):
        """
        Computes the shrinkage estimator for a covariance matrix using Ledoit and Wolf's (2003) method,
        utilizing pandas for efficient computation.

        Args:
        Y (pd.DataFrame): Input data frame representing samples.
        k (int, optional): Adjustment to the sample size. Defaults to None.

        Returns:
        pd.DataFrame: Shrinkage estimator of the covariance matrix.
        """
        N, p = Y.shape  # sample size and number of variables

        # Handle the demeaning process based on the condition of k
        if k is None or pd.isna(k):
            Y_demeaned = Y - Y.mean(axis=0)  # Demean Y directly using pandas operations
            k = 1
        else:
            Y_demeaned = Y

        n = N - k  # Adjusted sample size

        # Compute sample covariance matrix using pandas .cov() method
        # Note: pandas .cov() calculates the sample covariance matrix with N-1 in the denominator by default.
        # To adjust for our specific sample size (n), we'll manually scale the result.
        sample_cov = np.cov(Y_demeaned, rowvar=False) * ((N - 1) / n)

        # Compute shrinkage target (mean variance times identity matrix)
        mean_variance = np.mean(np.diag(sample_cov))
        target = np.eye(p) * mean_variance

        # Calculate pi_hat and gamma_hat using the formulas
        # Pi_hat calculation involves squared returns
        Y_squared = Y_demeaned ** 2
        sample_cov_squared = np.cov(Y_squared, rowvar=False) * ((N - 1) / n)

        pi_hat = np.sum(sample_cov_squared - np.square(sample_cov))

        # Gamma hat as the Frobenius norm squared difference between sample covariance and target
        gamma_hat = np.linalg.norm(sample_cov - target, 'fro') ** 2

        # Compute shrinkage intensity
        kappa_hat = pi_hat / gamma_hat  # Assuming rho parameters sum to 0 as before
        shrinkage = max(0, min(1, kappa_hat / n))

        # Compute shrinkage estimator
        sigma = shrinkage * target + (1 - shrinkage) * sample_cov
        sigma = 0.5 * (sigma + sigma.T)
        return sigma

    @staticmethod
    def auxiliary_data(X, beta):
        """
        Computes and returns an auxiliary weighted data matrix based on the input DataFrame X and decay factor beta.

        Parameters:
        - X (pd.DataFrame): The input data frame.
        - beta (float): The decay parameter, must be in the range (0, 1).

        Returns:
        - np.ndarray: A weighted version of the input data matrix.
        """
        if not (0 < beta < 1):
            raise ValueError("beta must be between 0 and 1")

        T = X.shape[0]
        j = np.arange(T)
        weights = T * (1 - beta) * beta ** j / (1 - beta ** T)
        W = np.sqrt(np.diag(weights[::-1]))
        return W @ X

    @staticmethod
    def _cross_validate_eigenvalues(X, e_val, n_fold, isotonic):
        """
        Returns cross-validated eigenvalues
        """
        T, N = X.shape
        # observations per fold
        obs_fold = T / n_fold

        # observations in small and observations in large K
        lowern = np.floor(obs_fold)
        uppern = np.ceil(obs_fold)
        lowern, uppern = int(lowern), int(uppern)

        # how many large and how many small K
        upperf = T - lowern * n_fold
        lowerf = n_fold - upperf

        np.random.seed(42)
        permut = np.random.permutation(T)
        eval_cv = 0.0

        for i in range(n_fold):
            cv_idx1 = np.zeros([T], dtype=bool)

            # get the indices for the eigenvalue estimation
            if i < lowerf:
                cv_idx1[permut[i * lowern: (i + 1) * lowern]] = True
            else:
                cv_idx1[permut[lowerf * lowern + (i - lowerf) * uppern:
                               lowerf * lowern + (i - lowerf + 1) * uppern]] = True

            # get the other indices for the variance estimation
            cv_idx2 = ~cv_idx1

            # compute train covariance matrix
            cov_train = np.cov(X.iloc[cv_idx2, :], rowvar=False)

            # perform eigendecomposition
            _, evec_train_cv = np.linalg.eigh(cov_train)

            # compute test covariance matrix
            cov_test = np.cov(X.iloc[cv_idx1, :], rowvar=False)

            # compute projections
            eval_cv += np.diag(evec_train_cv.T @ cov_test @ evec_train_cv / n_fold)

        # run isotonic regression
        if isotonic:
            eval_cv = IR().fit_transform(e_val, eval_cv)

        return eval_cv

    @staticmethod
    def auxiliary_data(X, beta):
        """
        Returns auxilliary data matrix
        """
        T = X.shape[0]
        j = np.arange(0, T)
        weights = T * (1 - beta) * beta ** j / (1 - beta ** T)
        W = np.diag(weights[::-1])
        return W ** (1 / 2) @ X

    def get_CV(self, X, n_fold=10, df=0, beta=None, scaling=True, isotonic=True):
        """
        Returns estimated covariance matrix and eigenvalues
        """
        T, N = X.shape
        T_eff = T - df

        # compute auxiliary data if beta value is given
        if beta:
            X = self.auxiliary_data(X, beta)

        # compute covariance
        E = np.cov(X, rowvar=False) * (T - 1) / T_eff

        # eigenvalue decomposition of the original data
        eval, evec = np.linalg.eigh(E)

        # generate CV eigenvalues
        eval_cv = self._cross_validate_eigenvalues(X, eval, n_fold, isotonic=isotonic)

        if scaling:
            scaling = np.trace(E) / np.sum(eval_cv)
            eval_cv = scaling * eval_cv
            assert math.isclose(np.trace(E), np.sum(eval_cv), rel_tol=0.00001)

        sigma = evec.dot(np.diag(eval_cv)).dot(evec.T)
        sigma = 0.5 * (sigma + sigma.T)

        return sigma

    def get_QIS(self, Y, k=None):
        # Pre-Conditions: Y is a valid pd.dataframe and optional arg- k which can be
        # None, np.nan or int
        # Post-Condition: Sigmahat dataframe is returned
        # Set df dimensions
        N, p = Y.shape

        # default setting
        if k is None or math.isnan(k):
            Y = Y.sub(Y.mean(axis=0), axis=1)  # demean
            k = 1
        # vars
        n = N - k  # adjust effective sample size
        c = p / n  # concentration ratio
        # Cov df: sample covariance matrix
        sample = np.cov(Y, rowvar=False) * (N - 1) / n
        sample = (sample + sample.T) / 2  # make symmetrical
        # Spectral decomp
        lambda1, u = np.linalg.eigh(sample)  # use Cholesky factorisation
        # based on hermitian matrix
        lambda1 = lambda1.real.clip(min=0)  # reset negative values to 0
        dfu = pd.DataFrame(u, columns=lambda1)  # create df with column names lambda
        # and values u
        dfu.sort_index(axis=1, inplace=True)  # sort df by column index
        lambda1 = dfu.columns  # recapture sorted lambda
        # COMPUTE Quadratic-Inverse Shrinkage estimator of the covariance matrix
        h = (min(c ** 2, 1 / c ** 2) ** 0.35) / p ** 0.35  # smoothing parameter
        invlambda = 1 / lambda1[max(1, p - n + 1) - 1:p]  # inverse of (non-null) eigenvalues
        dfl = pd.DataFrame()
        dfl['lambda'] = invlambda
        Lj = dfl[np.repeat(dfl.columns.values, min(p, n))]  # like 1/lambda_j
        Lj = pd.DataFrame(Lj.to_numpy())  # Reset column names
        Lj_i = Lj.subtract(Lj.T)  # like (1/lambda_j)-(1/lambda_i)
        theta = Lj.multiply(Lj_i).div(Lj_i.multiply(Lj_i).add(
            Lj.multiply(Lj) * h ** 2)).mean(axis=0)  # smoothed Stein shrinker
        Htheta = Lj.multiply(Lj * h).div(Lj_i.multiply(Lj_i).add(
            Lj.multiply(Lj) * h ** 2)).mean(axis=0)  # its conjugate
        Atheta2 = theta ** 2 + Htheta ** 2  # its squared amplitude
        if p <= n:  # case where sample covariance matrix is not singular
            delta = 1 / ((1 - c) ** 2 * invlambda + 2 * c * (
                        1 - c) * invlambda * theta + c ** 2 * invlambda * Atheta2)  # optimally shrunk eigenvalues
            delta = delta.to_numpy()
        else:
            delta0 = 1 / ((c - 1) * np.mean(invlambda.to_numpy()))  # shrinkage of null
            # eigenvalues
            delta = np.repeat(delta0, p - n)
            delta = np.concatenate((delta, 1 / (invlambda * Atheta2)), axis=None)
        deltaQIS = delta * (sum(lambda1) / sum(delta))  # preserve trace
        temp1 = dfu.to_numpy()
        temp2 = np.diag(deltaQIS)
        temp3 = dfu.T.to_numpy().conjugate()
        # reconstruct covariance matrix
        sigmahat = np.matmul(np.matmul(temp1, temp2), temp3)
        return sigmahat

    @staticmethod
    def _pav(y):
        """
        PAV uses the pair adjacent violators method to produce a monotonic
        smoothing of y
        translated from matlab by Sean Collins (2006) as part of the EMAP toolbox
        """
        y = np.asarray(y)
        assert y.ndim == 1
        n_samples = len(y)
        v = y.copy()
        lvls = np.arange(n_samples)
        lvlsets = np.c_[lvls, lvls]
        flag = 1
        while flag:
            deriv = np.diff(v)
            if np.all(deriv >= 0):
                break

            viol = np.where(deriv < 0)[0]
            start = lvlsets[viol[0], 0]
            last = lvlsets[viol[0] + 1, 1]
            s = 0
            n = last - start + 1
            for i in range(start, last + 1):
                s += v[i]

            val = s / n
            for i in range(start, last + 1):
                v[i] = val
                lvlsets[i, 0] = start
                lvlsets[i, 1] = last
        return v

    @staticmethod
    def get_NL(self, X, df=0):
        X = X.T
        N, T = X.shape
        T_eff = T - df
        C = np.cov(X, rowvar=False) * (T - 1) / T_eff

        eval, evec = np.linalg.eigh(C)
        eval, evec = eval[np.argsort(eval)], evec[:, np.argsort(eval)]

        eval = eval[max([1, N - T + 1]) - 1:N]
        L = np.tile(eval, (min([N, T]), 1)).T

        h = T ** (-0.35)

        Lt = (4 * (L ** 2) * (h ** 2) - (L - L.T) ** 2)

        ftilde = np.mean(np.sqrt(np.clip(Lt, 0, np.inf)) / (2 * np.pi * (L.T ** 2) * (h ** 2)), axis=1)

        Hftilde = np.mean(
            (np.sign(L - L.T) * np.sqrt(np.clip((L - L.T) ** 2 - 4 * (L.T ** 2) * (h ** 2), 0, np.inf)) - L + L.T) /
            (2 * np.pi * (L.T ** 2) * (h ** 2)), axis=1)

        if N <= T:
            dtilde = eval / ((np.pi * (N / float(T)) * eval * ftilde) ** 2 + (
                    1 - (N / float(T)) - np.pi * (N / float(T)) * eval * Hftilde) ** 2)

        else:
            Hftilde0 = (1 - np.sqrt(1 - 4 * h ** 2)) / (2 * np.pi * h ** 2) * np.mean(1. / eval)
            dtilde0 = 1. / (np.pi * (N - T) / float(T) * Hftilde0)
            dtilde1 = eval / (np.pi ** 2 * eval ** 2 * (ftilde ** 2 + Hftilde ** 2))
            dtilde = np.concatenate(([dtilde0] * (N - T), dtilde1))

        dhat = self._pav(dtilde)

        sigmahat = np.dot(evec, (np.tile(dhat, (N, 1)).T * evec.T))

        return sigmahat

    def get_QIS(self, Y, k=None):
        """
        Computes the Quadratic Inverse Shrinkage estimator of the covariance matrix for a given dataframe Y.

        Args:
            Y (pd.DataFrame): Input dataframe.
            k (None, np.nan, int): Adjustment factor for effective sample size, can be None, np.nan, or an integer.

        Returns:
            pd.DataFrame: The Quadratic Inverse Shrinkage estimator of the covariance matrix.
        """

        # Correct the description of dimensions
        N, p = Y.shape  # N: number of rows (samples), p: number of columns (features)

        # Default setting: demean Y if k is None or NaN, then set k to 1
        if k is None or math.isnan(k):
            Y = Y.sub(Y.mean(axis=0), axis=1)
            k = 1

        n = N - k  # Adjusted effective sample size
        c = p / n  # Concentration ratio

        # Use pandas .cov() method to compute sample covariance matrix
        sample_cov = np.cov(Y, rowvar=False) * ((N - 1) / n)  # Adjusting the normalization factor

        # Spectral decomposition
        eval, evec = np.linalg.eigh(sample_cov)
        eval = np.clip(eval, a_min=0, a_max=None)  # Ensure non-negative eigenvalues

        # Compute Quadratic Inverse Shrinkage estimator components
        h = (min(c ** 2, 1 / c ** 2) ** 0.35) / p ** 0.35
        valid_eval = eval[-min(p, n):]  # Consider non-null eigenvalues
        inverse_eval = 1 / valid_eval

        # Compute smoothed Stein shrinker and its conjugate
        Lj_matrix = np.tile(inverse_eval, (min(p, n), 1))
        Lj_diff = Lj_matrix - Lj_matrix.T
        theta = np.mean(Lj_matrix * Lj_diff / (Lj_diff ** 2 + h ** 2 * Lj_matrix ** 2), axis=0)
        Htheta = np.mean(h * Lj_matrix ** 2 / (Lj_diff ** 2 + h ** 2 * Lj_matrix ** 2), axis=0)
        Atheta2 = theta ** 2 + Htheta ** 2

        # Compute optimally shrunk eigenvalues
        if p <= n:
            delta = 1 / ((1 - c) ** 2 * inverse_eval + 2 * c * (1 - c) * theta + c ** 2 * Atheta2)
        else:
            delta0 = 1 / ((c - 1) * np.mean(inverse_eval))
            delta = np.concatenate((np.repeat(delta0, p - n), 1 / (inverse_eval * Atheta2)))

        # Preserve trace of the original covariance matrix
        deltaQIS = delta * (sum(eval) / sum(delta))

        # Reconstruct the covariance matrix
        Sigma = evec @ np.diag(deltaQIS) @ evec.T

        return Sigma
