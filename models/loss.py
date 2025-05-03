import torch


def kl_divergence(mu_z: torch.Tensor, logvar_z: torch.Tensor) -> torch.Tensor:
    """
    Calculate the KL divergence between the gaussian distribution N(mu_z, exp(logvar_z)) and N(0, I).
    Args:
        mu_z: [batch_size, z_dim]
        logvar_z: [batch_size, z_dim]
    """
    kl = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp(), dim=1)
    return kl.mean()


def l2_mean_error(action_pred: torch.Tensor, action: torch.Tensor):
    """
    L2 mean error between action and action_pred.
    Args:
        action_pred: [batch_size, action_horizon, action_dim]
        action: [batch_size, action_horizon, action_dim]
    """
    diff = action - action_pred
    return torch.mean(diff**2, dim=(0, 1, 2))


def l1_mean_error(action_pred: torch.Tensor, action: torch.Tensor):
    """
    L1 mean error between action and action_pred.
    Args:
        action_pred: [batch_size, action_horizon, action_dim]
        action: [batch_size, action_horizon, action_dim]
    """
    diff = action - action_pred
    return torch.mean(torch.abs(diff), dim=(0, 1, 2))
