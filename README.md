# CGCE · Contrastive Geometric Cross-Entropy Loss  
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)&nbsp;
![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-brightgreen)&nbsp;
![PyTorch ≥ 2.0](https://img.shields.io/badge/PyTorch-%E2%89%A52.0-orange)

> **CGCE** is a drop-in replacement for vanilla cross-entropy that **squeezes intra-class distances** and **stretches inter-class margins** directly in feature space.  
> By leveraging **learnable class prototypes** and **temperature-scaled similarities**, CGCE removes the need for a linear classification head—yielding faster convergence, higher accuracy and stronger robustness *without* adding any inference-time parameters.

---

## ✨ Highlights

|  | Cross-Entropy | **CGCE** |
|---|---|---|
| Decision boundary | Implicit (linear layer) | Explicit geometric similarity |
| Extra inference params | Yes (`W, b`) | **None** |
| Robustness | Standard | Better on long-tail / noisy / few-shot data |

---

## 📐 Loss Definition

Given  

| Symbol | Meaning |
|--------|---------|
| $\mathbf z\in\mathbb R^{d}$ | Sample feature (ℓ²-normalised) |
| $\mathbf u_k\in\mathbb R^{d}$ | Prototype of class $k$ (learnable & ℓ²-normalised) |
| $y$ | Ground-truth label |
| $\tau$ | Temperature (default 0.1) |

\[
\boxed{\;
\mathcal L_{\text{CGCE}}
= -\log
\frac{\exp\!\bigl(\langle\mathbf z,\mathbf u_{y}\rangle/\tau\bigr)}
{\displaystyle\sum_{k=0}^{K-1}\exp\!\bigl(\langle\mathbf z,\mathbf u_{k}\rangle/\tau\bigr)}
\;}
\]

* **Smaller $\tau$** → sharper margins; **larger $\tau$** → smoother optimisation.

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
