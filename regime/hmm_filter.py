# The core idea — add this as regime/hmm_filter.py
from hmmlearn.hmm import GaussianHMM


class RegimeFilter:
    """
    2-state Hidden Markov Model on realized variance.
    State 0 = low-vol (calm), State 1 = high-vol (stress).
    """

    def fit(self, returns):
        rv = (returns ** 2).rolling(5).mean().dropna().values.reshape(-1, 1)
        self.model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=100)
        self.model.fit(rv)
        return self

    def predict_regimes(self, returns):
        rv = (returns ** 2).rolling(5).mean().dropna().values.reshape(-1, 1)
        states = self.model.predict(rv)
        # Ensure state 0 = calm by checking mean variance per state
        if self.model.means_[0] > self.model.means_[1]:
            states = 1 - states  # flip labels
        return states