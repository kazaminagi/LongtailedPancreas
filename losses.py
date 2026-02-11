"""
EDL Losses for IBC-EDL
Supports soft label distributions from IBC's KNN-based label generation
"""
import torch
import torch.nn.functional as F


def relu_evidence(y):
    """Convert logits to evidence using ReLU"""
    return F.relu(y)


def kl_divergence(alpha, num_classes, device=None):
    """KL divergence between Dirichlet and uniform distribution"""
    if device is None:
        device = alpha.device
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


def edl_digamma_loss_soft(output, target_soft, epoch_num, num_classes, annealing_step, device=None):
    """
    EDL Digamma loss with soft labels
    
    Args:
        output: model logits [batch_size, num_classes]
        target_soft: soft label distribution [batch_size, num_classes]
        epoch_num: current epoch for annealing
        num_classes: number of classes
        annealing_step: steps for KL annealing
        device: computation device
    """
    if device is None:
        device = output.device
    
    target_soft = target_soft.to(device)
    evidence = relu_evidence(output)
    alpha = evidence + 1
    S = torch.sum(alpha, dim=1, keepdim=True)
    
    # EDL loss with soft labels
    A = torch.sum(target_soft * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    
    # KL divergence regularization
    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32, device=device),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32, device=device),
    )
    kl_alpha = (alpha - 1) * (1 - target_soft) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    
    return torch.mean(A + kl_div)


def edl_log_loss_soft(output, target_soft, epoch_num, num_classes, annealing_step, device=None):
    """EDL Log loss with soft labels"""
    if device is None:
        device = output.device
    
    target_soft = target_soft.to(device)
    evidence = torch.exp(torch.clamp(output, -10, 10))
    alpha = evidence + 1
    S = torch.sum(alpha, dim=1, keepdim=True)
    
    A = torch.sum(target_soft * (torch.log(S) - torch.log(alpha)), dim=1, keepdim=True)
    
    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32, device=device),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32, device=device),
    )
    kl_alpha = (alpha - 1) * (1 - target_soft) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    
    return torch.mean(A + kl_div)


def edl_mse_loss_soft(output, target_soft, epoch_num, num_classes, annealing_step, device=None):
    """EDL MSE loss with soft labels"""
    if device is None:
        device = output.device
    
    target_soft = target_soft.to(device)
    evidence = relu_evidence(output)
    alpha = evidence + 1
    S = torch.sum(alpha, dim=1, keepdim=True)
    
    # MSE loss
    loglikelihood_err = torch.sum((target_soft - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    
    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32, device=device),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32, device=device),
    )
    kl_alpha = (alpha - 1) * (1 - target_soft) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    
    return torch.mean(loglikelihood + kl_div)
