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
    mean = torch.mean(y, dtype=torch.float64)
    var = torch.mean((y-mean)**2)
    y = (y-mean) / var
    return y

def window_matrix(y: torch.Tensor, win_length: int, hop_length: int) -> torch.Tensor:
    n_wins = y.shape[-1] // hop_length if y.shape[-1] % hop_length == 0 else y.shape[-1] // hop_length + 1
    matrix = torch.zeros(y.shape[-1], n_wins)
    for win_index in range(n_wins):
        start = max(win_index * hop_length, 0)
        end = min(start + win_length, y.shape[-1])
        matrix[start:end, win_index] = 1
    
    return matrix