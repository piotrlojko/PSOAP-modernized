"""
A Metropolis-Hastings sampler for state-ful models within PSOAP.

Replaces the old ``StateSampler`` which depended on the deprecated HDF5
backend; chains are now saved as NumPy ``.npy`` files.
"""

import numpy as np
import logging


class StateSampler:
    """
    Minimal Metropolis-Hastings sampler compatible with the PSOAP sampling
    workflow.

    Args:
        lnprob (callable): log-probability function.
        p0 (1D np.array): starting parameter vector.
        cov (2D np.array): proposal covariance matrix.
        query_lnprob (callable, optional): called to get the current lnprob
            from a stateful model (e.g. when sub-process evaluations have
            modified the state since the last proposal).
        rejectfn (callable, optional): called when a proposal is rejected.
        acceptfn (callable, optional): called when a proposal is accepted.
        debug (bool): enable debug logging.
        outdir (str): output directory for saving intermediate chains.
    """

    def __init__(self, lnprob, p0, cov, query_lnprob=None, rejectfn=None,
                 acceptfn=None, debug=False, outdir=""):
        self.lnprob = lnprob
        self.p0 = np.asarray(p0, dtype=np.float64)
        self.dim = len(self.p0)
        self.cov = np.asarray(cov, dtype=np.float64)
        self.query_lnprob = query_lnprob
        self.rejectfn = rejectfn
        self.acceptfn = acceptfn
        self.outdir = outdir
        self.debug = debug

        self._rng = np.random.default_rng()
        self.iterations = 0
        self.naccepted = 0
        self._chain = np.empty((0, self.dim))
        self._lnprob = np.empty(0)

        self.logger = logging.getLogger(self.__class__.__name__)
        if debug:
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)

    def reset(self):
        self.iterations = 0
        self.naccepted = 0
        self._chain = np.empty((0, self.dim))
        self._lnprob = np.empty(0)

    @property
    def flatchain(self):
        return self._chain

    @property
    def lnprobability(self):
        return self._lnprob

    @property
    def acceptance_fraction(self):
        if self.iterations == 0:
            return 0.0
        return self.naccepted / self.iterations

    def sample(self, p0, lnprob0=None, thin=1, storechain=True,
               iterations=1, incremental_save=0):
        """
        Advance the chain for ``iterations`` steps.

        Yields:
            (p, lnprob, None): current position, log-probability, and
                placeholder for RNG state.
        """
        p = np.asarray(p0, dtype=np.float64)

        if lnprob0 is None:
            if self.query_lnprob is not None:
                lnprob0 = self.query_lnprob()
            if lnprob0 is None:
                lnprob0 = self.lnprob(p)

        if storechain:
            N = int(iterations / thin)
            self._chain = np.concatenate(
                (self._chain, np.zeros((N, self.dim))), axis=0)
            self._lnprob = np.append(self._lnprob, np.zeros(N))

        i0 = self.iterations

        for i in range(int(iterations)):
            self.iterations += 1

            if self.query_lnprob is not None:
                lnprob0 = self.query_lnprob()
                self.logger.debug("Queried lnprob: {}".format(lnprob0))

            if self.dim == 1:
                q = self._rng.normal(loc=p[0], scale=np.sqrt(self.cov[0, 0]),
                                     size=(1,))
            else:
                q = self._rng.multivariate_normal(p, self.cov)

            newlnprob = self.lnprob(q)
            diff = newlnprob - lnprob0
            self.logger.debug("old lnprob: {}, proposed: {}".format(
                lnprob0, newlnprob))

            if diff < 0:
                diff = np.exp(diff) - self._rng.random()

            if diff >= 0:
                p = q
                lnprob0 = newlnprob
                self.naccepted += 1
                if self.acceptfn is not None:
                    self.acceptfn()
            else:
                if self.rejectfn is not None:
                    self.rejectfn()

            if storechain and i % thin == 0:
                ind = i0 + int(i / thin)
                self._chain[ind, :] = p
                self._lnprob[ind] = lnprob0

            if incremental_save and (((i + 1) % incremental_save) == 0) and i > 0:
                np.save(self.outdir + 'chain_backup.npy', self._chain)

            yield p, lnprob0, None

    def write(self, fname="chain.npy"):
        """Save the flat chain to a NumPy ``.npy`` file."""
        np.save(self.outdir + fname, self._chain)
