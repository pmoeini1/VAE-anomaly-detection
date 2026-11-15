from loss_functions import gaussian_log_prob
import torch

@torch.no_grad()
def anomaly_score(model, x, L=20):
    model.eval()
    mu, logvar = model.encode(x)
    log_probs = []
    for _ in range(L):
        z = model.reparameterize(mu, logvar)
        x_mu, x_logvar = model.decode(z)
        log_probs.append(gaussian_log_prob(x, x_mu, x_logvar))
    return -torch.stack(log_probs).mean(dim=0)