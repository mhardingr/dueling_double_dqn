from scipy.stats import rv_continuous, norm
import numpy as np 

class P_gen(rv_continuous):
    """Represents the PDF p(x) = 1/2*(1+x), -1<=x<=1"""
    def __init__(self, seed=None):
        super(P_gen, self).__init__(a=-1, b=1, seed=seed)
    def _pdf(self, x):
        return .5*(1+x)

class q_poly_gen(rv_continuous):
    """Represents the PDF q(x) = 15/16*x**2*(1+x)**2, -1<=x<=1 """
    def __init__(self, seed=None):
        super(q_poly_gen, self).__init__(a=-1, b=1, seed=seed)
    def _pdf(self, x):
        return 15.0/16.0*x**2*(1+x)**2

def get_expected_value_and_variance(pdf, f_x, n_samples, seed=None):
    samples = pdf.rvs(size=n_samples, random_state=seed)
    vfunc = np.vectorize(f_x)
    f_x_samples = vfunc(samples)
    sample_mean = np.mean(f_x_samples)
    sample_variance = np.mean( np.square((f_x_samples - sample_mean)) )

    return sample_mean, sample_variance

def get_exp_var_using_importance_sampling(p_gen, q_gen, f_x, n_samples, 
                                            q_loc=0, q_scale=1, 
                                            seed=None):
    # Generate samples z from q_pdf, then
    ## Find stats on rational function = p_pdf(z)/q_pdf(z)*f_x(z)
    samples = q_gen.rvs(size=n_samples, random_state=seed,
                        loc=q_loc, scale=q_scale)
    vfunc = np.vectorize(lambda z: p_gen.pdf(z) \
                                     / q_gen.pdf(z, loc=q_loc, scale=q_scale) \
                                        * f_x(z))
    impt_samples = vfunc(samples)
    sample_mean = np.mean(impt_samples)
    sample_variance = np.mean( np.square((impt_samples - sample_mean)) )

    return sample_mean, sample_variance

def get_exp_var_weighted_importance_sampling(p_gen, q_gen, f_x, n_samples, 
                                            q_loc=0, q_scale=1, 
                                            seed=None):
    # Generate samples z from q_pdf, then
    ## Find stats on the weight-normalized rational function 
    ## = (p_pdf(z)/q_pdf(z)*f_x(z))/(sum(p_pdf(z)/q_pdf(z))
    samples = q_gen.rvs(size=n_samples, random_state=seed,
                        loc=q_loc, scale=q_scale)
    # Compute weights using vectorize over samples from q (assumed normalized)
    weightsfunc = np.vectorize(lambda z: p_gen.pdf(z) / \
                                        q_gen.pdf(z,loc=q_loc, scale=q_scale))
    weights = weightsfunc(samples)
    # Compute weighted importance samples of f_x
    norm_denom = np.sum(weights)
    vfunc = np.vectorize(lambda z: p_gen.pdf(z) \
                                     / q_gen.pdf(z, loc=q_loc, scale=q_scale) \
                                        * f_x(z) \
                                        / norm_denom)
    weighted_impt_samples = vfunc(samples)
    sample_mean = np.sum(weighted_impt_samples)
    sample_variance = np.mean( np.square((weighted_impt_samples - sample_mean)) )

    return sample_mean, sample_variance

if __name__ == "__main__":
    print "Initializing ..."
    """ 
    seed = 10703
    p_x = P_gen()
    f_x = lambda x: 1.5*x**2*(1+x)
    exp10, var10 = get_expected_value_and_variance(p_x, f_x, 10, seed=seed)
    print "For 10 samples, expected value and variance:", exp10, var10
    exp1000, var1000 = get_expected_value_and_variance(p_x, f_x, 1000, seed=seed)
    print "For 1000 samples, expected value and variance:", exp1000, var1000
    exp10k, var10k = get_expected_value_and_variance(p_x, f_x, 10000, seed=seed)
    print "For 10k samples, expected value and variance:", exp10k, var10k
    """
    """
    print "On to importance sampling ..."
    p = P_gen()
    q = norm
    f_x = lambda x: 1.5*x**2*(1+x)
    print "Q(x) = norm(3,1):"
    exp10, var10 = get_exp_var_using_importance_sampling(p, q, f_x, 10,
            q_loc=3, q_scale=1, seed=seed)
    print "For 10 samples, expected value and variance:", exp10, var10
    exp1000, var1000 = get_exp_var_using_importance_sampling(p, q, f_x, 1000,
            q_loc=3, q_scale=1, seed=seed)
    print "For 1000 samples, expected value and variance:", exp1000, var1000
    exp10k, var10k = get_exp_var_using_importance_sampling(p, q, f_x, 10000,
            q_loc=3, q_scale=1, seed=seed)    
    print "For 10k samples, expected value and variance:", exp10k, var10k

    print "Q(x) = norm(0,1):"
    exp10, var10 = get_exp_var_using_importance_sampling(p, q, f_x, 10,
            q_loc=0, q_scale=1, seed=seed)
    print "For 10 samples, expected value and variance:", exp10, var10
    exp1000, var1000 = get_exp_var_using_importance_sampling(p, q, f_x, 1000,
            q_loc=0, q_scale=1, seed=seed)
    print "For 1000 samples, expected value and variance:", exp1000, var1000
    exp10k, var10k = get_exp_var_using_importance_sampling(p, q, f_x, 10000,
            q_loc=0, q_scale=1, seed=seed)    
    print "For 10k samples, expected value and variance:", exp10k, var10k

    q = q_poly_gen()
    print "Q(x) = 15/16*x^2*(1+x)^2:"
    exp10, var10 = get_exp_var_using_importance_sampling(p, q, f_x, 10,
                                                            seed=seed)
    print "For 10 samples, expected value and variance:", exp10, var10
    exp1000, var1000 = get_exp_var_using_importance_sampling(p, q, f_x, 1000,
                                                            seed=seed)
    print "For 1000 samples, expected value and variance:", exp1000, var1000
    exp10k, var10k = get_exp_var_using_importance_sampling(p, q, f_x, 10000,
                                                            seed=seed)    
    print "For 10k samples, expected value and variance:", exp10k, var10k
    """

    print "On to weighted importance sampling ..."
    p = P_gen()
    q = norm  # Note: normal distribution is normalized, so it's Z_q = 1
    f_x = lambda x: 1.5*x**2*(1+x)
    seed = 10703
    print "Q(x) = norm(3,1):"
    exp10, var10 = get_exp_var_weighted_importance_sampling(p, q, f_x, 10,
            q_loc=3, q_scale=1, seed=seed)
    print "For 10 samples, expected value and variance:", exp10, var10
    exp1000, var1000 = get_exp_var_weighted_importance_sampling(p, q, f_x, 1000,
            q_loc=3, q_scale=1, seed=seed)
    print "For 1000 samples, expected value and variance:", exp1000, var1000
    exp10k, var10k = get_exp_var_weighted_importance_sampling(p, q, f_x, 10000,
            q_loc=3, q_scale=1, seed=seed)    
    print "For 10k samples, expected value and variance:", exp10k, var10k

    print "Q(x) = norm(0,1):"
    exp10, var10 = get_exp_var_weighted_importance_sampling(p, q, f_x, 10,
            q_loc=0, q_scale=1, seed=seed)
    print "For 10 samples, expected value and variance:", exp10, var10
    exp1000, var1000 = get_exp_var_weighted_importance_sampling(p, q, f_x, 1000,
            q_loc=0, q_scale=1, seed=seed)
    print "For 1000 samples, expected value and variance:", exp1000, var1000
    exp10k, var10k = get_exp_var_weighted_importance_sampling(p, q, f_x, 10000,
            q_loc=0, q_scale=1, seed=seed)    
    print "For 10k samples, expected value and variance:", exp10k, var10k

    # Note: this polynomial q is normalized within domain -1<=x<=1, Z_q = 1
    q = q_poly_gen()      
    print "Q(x) = 15/16*x^2*(1+x)^2:"
    exp10, var10 = get_exp_var_weighted_importance_sampling(p, q, f_x, 10,
                                                            seed=seed)
    print "For 10 samples, expected value and variance:", exp10, var10
    exp1000, var1000 = get_exp_var_weighted_importance_sampling(p, q, f_x, 1000,
                                                            seed=seed)
    print "For 1000 samples, expected value and variance:", exp1000, var1000
    exp10k, var10k = get_exp_var_weighted_importance_sampling(p, q, f_x, 10000,
                                                            seed=seed)    
    print "For 10k samples, expected value and variance:", exp10k, var10k

