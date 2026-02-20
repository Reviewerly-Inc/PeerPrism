import torch


def histcounts(data, epsilon, min_=-1, max_=1):
    data = data.float()
    hist = torch.histc(data, bins=epsilon, min=min_, max=max_)
    statistical_probabilities_sequence = hist / torch.sum(hist)
    return hist, statistical_probabilities_sequence


def DE(statistical_probabilities_sequence, epsilon):
    DE_value = -1 / torch.log(torch.tensor(epsilon)) * torch.nansum(
        statistical_probabilities_sequence * torch.log(statistical_probabilities_sequence), dim=0
    )
    return DE_value


def calculate_DE(ori_data, embed_size, epsilon):
    orbits = ori_data.unfold(1, embed_size, 1)
    orbits_cosine_similarity_sequence = torch.nn.functional.cosine_similarity(
        orbits[:, :-1], orbits[:, 1:], dim=-1
    )
    batched_1 = torch.vmap(histcounts, in_dims=-1, out_dims=1)
    hist, statistical_probabilities_sequence = batched_1(
        orbits_cosine_similarity_sequence, epsilon=epsilon
    )
    DE_value = DE(statistical_probabilities_sequence, epsilon)
    return DE_value


def get_tau_scale_DE(ori_data, embed_size, epsilon, tau):
    windows = ori_data.unfold(1, tau, 1)
    tau_scale_sequence = torch.mean(windows, dim=3)
    de = calculate_DE(tau_scale_sequence, embed_size, epsilon)
    return de


def get_tau_multiscale_DE(ori_data, embed_size, epsilon, tau_prime):
    mde = []
    for temp_tau in range(1, tau_prime + 1):
        value = get_tau_scale_DE(ori_data, embed_size, epsilon, temp_tau)
        mde.append(value)
    mde = torch.stack(mde, dim=0)
    std_mde = torch.std(mde, dim=0)
    return std_mde
