import numpy as np
from skimage.metrics import structural_similarity as ssim

def mse01(a, b):
    return float(np.mean((a - b) ** 2))

def ssim01(a, b):
    a = np.clip(a.squeeze(), 0, 1)
    b = np.clip(b.squeeze(), 0, 1)
    return float(ssim(a, b, data_range=1.0))

def composite_fitness(y_pred, y_true, n_params, alpha=0.7, beta=0.3, lam=0.02):
    """
    Fitness thấp hơn = tốt hơn.
    alpha: trọng số MSE
    beta: trọng số (1-SSIM)
    lam : phạt theo số tham số (triệu tham số)
    """
    m = mse01(y_pred, y_true)
    s = ssim01(y_pred, y_true)
    return alpha*m + beta*(1.0 - s) + lam*(n_params/1e6)
