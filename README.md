# CGCE · Contrastive Geometric Cross-Entropy Loss  
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)&nbsp;
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-brightgreen)&nbsp;
![PyTorch ≥ 2.0](https://img.shields.io/badge/PyTorch-%E2%89%A52.0-orange)

## Overview  
Contrastive Geometric Cross-Entropy (CGCE) is a loss function that unifies cross‐entropy classification with contrastive, geometric margin optimization. Instead of computing logits via a linear classifier, CGCE measures cosine similarities between normalized example embeddings and learned label embeddings, then applies a temperature‐scaled cross‐entropy over the resulting similarity matrix.

## Abstract  
We introduce CGCE, a loss that replaces the final linear layer’s logits with a contrastive similarity matrix between example features and learnable class prototypes. By optimizing geometric margins in embedding space and jointly learning label prototypes, CGCE yields better calibrated decision boundaries and faster convergence. We validate CGCE on CIFAR‐10, CIFAR‐100 and ImageNet, showing consistent improvements in top‐1 accuracy and robustness under label noise.

## Features  
- **Contrastive margin optimization**  
  Maximizes inter‐class angular separation and minimizes intra‐class variance.  
- **Learnable label prototypes**  
  Each class is represented by a trainable embedding, initialized via class‐mean features.  
- **Temperature scaling**  
  Controls softness of the similarity distribution.  
- **Plug‐and‐play**  
  Drop‐in replacement for standard `CrossEntropyLoss`.

### Minimal PyTorch Implementation

```python
import torch, torch.nn as nn, torch.nn.functional as F

class CGCE(nn.Module):
    """Contrastive Geometric Cross-Entropy loss"""
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.tau = temperature
        self.ce  = nn.CrossEntropyLoss()

    def forward(self, feats: torch.Tensor,
                proto: torch.Tensor,
                labels: torch.Tensor):
        logits = feats @ proto.T / self.tau      # [B, K] similarities
        return self.ce(logits, labels)

class LabelEmbeddings(nn.Module):
    """Learnable class prototypes U ∈ ℝ^{K×d}"""
    def __init__(self, num_classes: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(num_classes, dim)
        nn.init.normal_(self.emb.weight, 0.0, 0.01)

    def forward(self) -> torch.Tensor:
        return F.normalize(self.emb.weight, dim=1)  # ℓ²-norm
