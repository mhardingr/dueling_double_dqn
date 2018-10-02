from scipy.stats import rv_continuous
import numpy as np 

class P_gen(rv_continuous):
    """Represents the PDF p(x) = 1/2*(1+x), -1<=x<=1"""
    def __init__(self, seed=None):
        super(P_gen, self).__init__(a=-1, b=1, seed=seed)
    def _pdf(self, x):
        return .5*(1+x)

def get_expected_value_and_variance(pdf, f_x, n_samples, seed=None):
    samples = pdf.rvs(size=n_samples, random_state=seed)
    vfunc = np.vectorize(f_x)
    f_x_samples = vfunc(samples)
    sample_mean = np.mean(f_x_samples)
    sample_variance = np.mean( np.square((f_x_samples - sample_mean)) )

    return sample_mean, sample_variance


if __name__ == "__main__":
    print "Initializing ..."
    seed = 10703
    p_x = P_gen(seed=seed)
    f_x = lambda x: 1.5*x**2*(1+x)
    exp10, var10 = get_expected_value_and_variance(p_x, f_x, 10, seed=seed)
    print "For 10 samples, expected value and variance:", exp10, var10
    exp1000, var1000 = get_expected_value_and_variance(p_x, f_x, 1000, seed=seed)
    print "For 1000 samples, expected value and variance:", exp1000, var1000
    exp10k, var10k = get_expected_value_and_variance(p_x, f_x, 10000, seed=seed)
    print "For 10k samples, expected value and variance:", exp10k, var10k

