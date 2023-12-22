import torch

def feature_mean(feature: torch.Tensor) -> torch.Tensor:
    """
    特徴量から各次元ごとの平均を求める
    
    Parameters
    ----------
    y : torch.Tensor
        対象の音楽
    
    Returns
    ----------
    feature : torch.Tensor
        変換後の特徴量
    """

    return torch.mean(feature, dim=-1)


def feature_var(feature: torch.Tensor) -> torch.Tensor:
    """
    特徴量から各次元ごとの分散を求める
    
    Parameters
    ----------
    y : torch.Tensor
        対象の音楽
    
    Returns
    ----------
    feature : torch.Tensor
        変換後の特徴量
    """

    return torch.var(feature, dim=-1)


def normalize(y: torch.Tensor) -> torch.Tensor:
    batch_max = torch.sum(y, dim=-1)
    y /= batch_max
    return y