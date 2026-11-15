import math
# log p(x|z) summed over window
def gaussian_log_prob(x, mu, logvar):
    return -0.5*(math.log(2*math.pi) + logvar + (x-mu) ** 2 / logvar.exp()).sum(dim=1)

def neg_elbo(x, x_mu, x_logvar, mu, logvar):
    recon = -gaussian_log_prob(x, x_mu, x_logvar)
    kl = -0.5 * (1 + logvar - mu**2 - logvar.exp()).sum(dim=1)
    return (recon+kl).mean()
