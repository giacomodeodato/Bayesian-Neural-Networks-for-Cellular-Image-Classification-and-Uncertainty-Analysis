import numpy as np

def confidence_score(p : np.array) -> np.array:
    """Returns confidence score of a set of predictive samples.

    The confidence score is computed as: 1 - 2 * sqrt(u), where u is
    the sum aleatoric and epistemic uncertainties of the sample as
    defined by [1].

    Parameters
    ----------
    p : np.array()
        Array of predictive samples of shape
        (n_predictive_samples, n_data_samples, n_classes)

    Returns
    -------
    np.array of shape (n_data_samples,)
        Confidence score for each data point provided.

    References
    ----------
        [1] Kendall, A., & Gal, Y. (2017). What uncertainties do we 
        need in bayesian deep learning for computer vision?. arXiv 
        preprint arXiv:1703.04977.

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