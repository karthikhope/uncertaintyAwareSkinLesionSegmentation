import torch


def predictive_entropy(mean_prob, eps=1e-7):
    """
    Total uncertainty: H(p̄)
    Entropy of the mean predictive distribution.
    """
    return -(
        mean_prob * torch.log(mean_prob + eps)
        + (1 - mean_prob) * torch.log(1 - mean_prob + eps)
    )


def expected_entropy(samples, eps=1e-7):
    """
    Aleatoric-like component: E[H(p_t)]
    Average entropy across T stochastic forward passes.
    samples: [T, B, 1, H, W]
    """
    ent = -(
        samples * torch.log(samples + eps)
        + (1 - samples) * torch.log(1 - samples + eps)
    )
    return ent.mean(dim=0)


def mutual_information(mean_prob, samples):
    """
    Epistemic uncertainty proxy: MI = H(p̄) - E[H(p_t)]
    High MI → model is uncertain due to lack of knowledge.
    """
    pe = predictive_entropy(mean_prob)
    ee = expected_entropy(samples)
    return pe - ee
