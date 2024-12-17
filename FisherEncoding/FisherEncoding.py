import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.mixture import GaussianMixture
import numpy as np


# Step 1: Train a Gaussian Mixture Model (GMM)
def train_gmm(descriptors, n_components=5):
    """
    Train a GMM on input descriptors.
    :param descriptors: np.ndarray of shape (n_samples, feature_dim)
    :param n_components: Number of GMM components
    :return: Trained GMM model
    """
    gmm = GaussianMixture(n_components=n_components, covariance_type='diag', random_state=42)
    gmm.fit(descriptors)
    return gmm


# Step 2: Compute Fisher Vector Encoding
def fisher_vector(descriptors, gmm):
    """
    Compute the Fisher Vector for the given descriptors.
    :param descriptors: np.ndarray of shape (n_samples, feature_dim)
    :param gmm: Trained GMM model
    :return: Fisher Vector as a PyTorch Tensor
    """
    # Retrieve GMM parameters
    means = torch.tensor(gmm.means_, dtype=torch.float32)  # Shape: (n_components, feature_dim)
    covariances = torch.tensor(gmm.covariances_, dtype=torch.float32)  # Shape: (n_components, feature_dim)
    weights = torch.tensor(gmm.weights_, dtype=torch.float32)  # Shape: (n_components)

    # Convert descriptors to tensor
    descriptors = torch.tensor(descriptors, dtype=torch.float32)  # Shape: (n_samples, feature_dim)

    # Compute posterior probabilities (responsibilities)
    log_prob = torch.tensor(gmm._estimate_log_prob(descriptors))
    log_resp = log_prob + torch.log(weights)
    log_resp -= torch.logsumexp(log_resp, dim=1, keepdim=True)
    responsibilities = torch.exp(log_resp)

    # Compute sufficient statistics
    n_samples, feature_dim = descriptors.shape
    n_components = gmm.n_components
    fisher_vector_means = torch.zeros((n_components, feature_dim))
    fisher_vector_covs = torch.zeros((n_components, feature_dim))

    for k in range(n_components):
        gamma_k = responsibilities[:, k]  # Shape: (n_samples,)
        gamma_sum = torch.sum(gamma_k)  # Scalar

        # Mean statistics
        diff = descriptors - means[k]  # Shape: (n_samples, feature_dim)
        fisher_vector_means[k] = torch.sum(gamma_k.unsqueeze(1) * diff, dim=0) / torch.sqrt(weights[k])

        # Covariance statistics
        fisher_vector_covs[k] = torch.sum(gamma_k.unsqueeze(1) * (diff ** 2 / covariances[k] - 1), dim=0) / torch.sqrt(
            2 * weights[k])

    # Flatten and concatenate both parts of the Fisher Vector
    fisher_vector = torch.cat((fisher_vector_means.flatten(), fisher_vector_covs.flatten()))

    # Normalize Fisher Vector
    fisher_vector = nn.functional.normalize(fisher_vector, p=2, dim=0)
    return fisher_vector


# Step 3: Example usage
if __name__ == "__main__":
    # Simulated descriptors (e.g., features from an image)
    n_samples = 1000  # Number of descriptors
    feature_dim = 128  # Dimension of each descriptor
    descriptors = np.random.rand(n_samples, feature_dim).astype(np.float32)

    # Train GMM
    n_components = 5  # Number of Gaussian components
    gmm = train_gmm(descriptors, n_components=n_components)
    print("GMM training completed.")

    # Compute Fisher Vector
    fisher_vector_result = fisher_vector(descriptors, gmm)
    print("Fisher Vector Shape:", fisher_vector_result.shape)
    print("Fisher Vector:", fisher_vector_result)
