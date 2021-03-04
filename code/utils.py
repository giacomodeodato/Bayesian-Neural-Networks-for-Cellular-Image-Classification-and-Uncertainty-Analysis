import numpy as np

def confidence_score(p):
    """
    p: np.array of shape (n_predictive_samples, n_data_samples, n_classes)
    """
    
    # compute sample mean
    p_mean = np.mean(p, axis=0)
    
    # compute uncertainty estimate
    aleatoric = np.mean(p - np.square(p), axis=0)
    epistemic = np.mean(np.square(p - p_mean), axis=0)
    u = aleatoric + epistemic
    
    # select uncertainty corresponding to the predicted class
    u = u[np.arange(len(p_mean)), np.argmax(p_mean, axis=1)]
    
    # return confidence score
    return 1 - 2 * np.sqrt(u)