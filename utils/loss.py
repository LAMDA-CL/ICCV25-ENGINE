from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    """ SimCLR loss @SimCLR
    Adapted from:
    https://github.com/ysharma1126/ssl_identifiability/blob/master/main_3dident.py
    """
    def __init__(self, tau: float = 0.5) -> None:
        super().__init__()
        self._tau = tau
        assert self._tau != 0
        self._metric = nn.CosineSimilarity(dim=-1)
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        sim_xx = self._metric(x.unsqueeze(-2), x.unsqueeze(-3)) / self._tau
        sim_yy = self._metric(y.unsqueeze(-2), y.unsqueeze(-3)) / self._tau
        sim_xy = self._metric(x.unsqueeze(-2), y.unsqueeze(-3)) / self._tau

        n = sim_xy.shape[-1]
        sim_xx[..., range(n), range(n)] = float("-inf")
        sim_yy[..., range(n), range(n)] = float("-inf")
        scores1 = torch.cat([sim_xy, sim_xx], dim=-1)    
        scores2 = torch.cat([sim_yy, sim_xy.transpose(-1,-2)], dim=-1)     
        scores = torch.cat([scores1, scores2], dim=-2)  
        targets = torch.arange(2 * n, dtype=torch.long, device=scores.device)
        total_loss = self.criterion(scores, targets)
        return total_loss
    
class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
    
    def forward(self, x, y):
        cosine_sim = self.cosine_similarity(x, y)
        loss = 1 - cosine_sim.mean()
        return loss

def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))
 
def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0

