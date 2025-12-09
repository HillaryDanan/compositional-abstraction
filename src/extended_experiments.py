"""
Extended Compositional Abstraction Experiments
===============================================

Three studies testing the robustness and boundaries of the factorization finding:

Study 1: Holdout Sweep
    - Same 3×3 domain
    - Vary number of held-out compositions: 2, 3, 4, 5
    - Question: Does the effect hold as generalization demand increases?

Study 2: Harder Domain
    - 4 colors × 4 shapes × 3 sizes = 48 compositions
    - Hold out 12 (25%)
    - Question: Does factorization scale to more factors?

Study 3: Partial Factorization
    - Vary degree of encoder sharing: 0% (fully factorized) to 100% (holistic)
    - Question: Is compositionality binary or graded?

Author: Hillary Danan, PhD
Date: December 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import random
from pathlib import Path
from datetime import datetime


# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 42
EPOCHS = 1000
LR = 0.01
N_RUNS = 10

# Study 1 & 3: Original domain
COLORS_3 = ["red", "blue", "green"]
SHAPES_3 = ["circle", "square", "triangle"]

# Study 2: Extended domain
COLORS_4 = ["red", "blue", "green", "yellow"]
SHAPES_4 = ["circle", "square", "triangle", "pentagon"]
SIZES_3 = ["small", "medium", "large"]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def create_compositional_holdout(all_compositions, n_holdout, n_factors, seed=SEED):
    """
    Create a valid compositional holdout.
    
    Valid = each factor level appears in training at least once.
    This ensures test items are truly NOVEL COMPOSITIONS, not novel factors.
    
    Args:
        all_compositions: list of tuples, each tuple has n_factors elements
        n_holdout: number of compositions to hold out
        n_factors: number of factors (2 for color+shape, 3 for color+shape+size)
        seed: random seed
    """
    random.seed(seed)
    
    # Extract unique levels for each factor
    factor_levels = [set() for _ in range(n_factors)]
    for item in all_compositions:
        for i, level in enumerate(item):
            factor_levels[i].add(level)
    
    # Try random holdouts until we find a valid one
    for attempt in range(1000):
        holdout = random.sample(all_compositions, n_holdout)
        train = [c for c in all_compositions if c not in holdout]
        
        # Check each factor appears in training
        valid = True
        for i in range(n_factors):
            train_levels = set(item[i] for item in train)
            if train_levels != factor_levels[i]:
                valid = False
                break
        
        if valid:
            return train, holdout
    
    raise ValueError(f"Could not find valid holdout of size {n_holdout}")


# =============================================================================
# MODELS: TWO-FACTOR (Study 1 & 3)
# =============================================================================

class HolisticAE2(nn.Module):
    """Holistic autoencoder for 2-factor domain."""
    
    def __init__(self, n_colors, n_shapes, embed_dim=4, hidden_dim=8, latent_dim=4):
        super().__init__()
        self.n_colors = n_colors
        self.n_shapes = n_shapes
        
        self.embed = nn.Embedding(n_colors + n_shapes, embed_dim)
        
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU()
        )
        self.color_head = nn.Linear(hidden_dim, n_colors)
        self.shape_head = nn.Linear(hidden_dim, n_shapes)
    
    def forward(self, color_idx, shape_idx):
        e_color = self.embed(color_idx)
        e_shape = self.embed(shape_idx + self.n_colors)
        x = torch.cat([e_color, e_shape], dim=-1)
        z = self.encoder(x)
        h = self.decoder(z)
        return self.color_head(h), self.shape_head(h)
    
    def loss(self, color_idx, shape_idx):
        color_logits, shape_logits = self.forward(color_idx, shape_idx)
        return (F.cross_entropy(color_logits, color_idx) + 
                F.cross_entropy(shape_logits, shape_idx))
    
    def predict(self, color_idx, shape_idx):
        self.eval()
        with torch.no_grad():
            color_logits, shape_logits = self.forward(color_idx, shape_idx)
        self.train()
        return color_logits.argmax(-1), shape_logits.argmax(-1)


class FactorizedAE2(nn.Module):
    """Factorized autoencoder for 2-factor domain."""
    
    def __init__(self, n_colors, n_shapes, embed_dim=4, hidden_dim=8, latent_dim=4):
        super().__init__()
        
        self.color_embed = nn.Embedding(n_colors, embed_dim)
        self.shape_embed = nn.Embedding(n_shapes, embed_dim)
        
        half_latent = latent_dim // 2
        
        self.color_encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, half_latent)
        )
        self.shape_encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, half_latent)
        )
        
        self.color_decoder = nn.Sequential(
            nn.Linear(half_latent, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_colors)
        )
        self.shape_decoder = nn.Sequential(
            nn.Linear(half_latent, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_shapes)
        )
    
    def forward(self, color_idx, shape_idx):
        z_color = self.color_encoder(self.color_embed(color_idx))
        z_shape = self.shape_encoder(self.shape_embed(shape_idx))
        return self.color_decoder(z_color), self.shape_decoder(z_shape)
    
    def loss(self, color_idx, shape_idx):
        color_logits, shape_logits = self.forward(color_idx, shape_idx)
        return (F.cross_entropy(color_logits, color_idx) + 
                F.cross_entropy(shape_logits, shape_idx))
    
    def predict(self, color_idx, shape_idx):
        self.eval()
        with torch.no_grad():
            color_logits, shape_logits = self.forward(color_idx, shape_idx)
        self.train()
        return color_logits.argmax(-1), shape_logits.argmax(-1)


# =============================================================================
# MODELS: THREE-FACTOR (Study 2)
# =============================================================================

class HolisticAE3(nn.Module):
    """Holistic autoencoder for 3-factor domain."""
    
    def __init__(self, n_colors, n_shapes, n_sizes, embed_dim=4, hidden_dim=16, latent_dim=6):
        super().__init__()
        self.n_colors = n_colors
        self.n_shapes = n_shapes
        self.n_sizes = n_sizes
        
        total_vocab = n_colors + n_shapes + n_sizes
        self.embed = nn.Embedding(total_vocab, embed_dim)
        
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU()
        )
        self.color_head = nn.Linear(hidden_dim, n_colors)
        self.shape_head = nn.Linear(hidden_dim, n_shapes)
        self.size_head = nn.Linear(hidden_dim, n_sizes)
    
    def forward(self, color_idx, shape_idx, size_idx):
        e_color = self.embed(color_idx)
        e_shape = self.embed(shape_idx + self.n_colors)
        e_size = self.embed(size_idx + self.n_colors + self.n_shapes)
        x = torch.cat([e_color, e_shape, e_size], dim=-1)
        z = self.encoder(x)
        h = self.decoder(z)
        return self.color_head(h), self.shape_head(h), self.size_head(h)
    
    def loss(self, color_idx, shape_idx, size_idx):
        c, s, z = self.forward(color_idx, shape_idx, size_idx)
        return (F.cross_entropy(c, color_idx) + 
                F.cross_entropy(s, shape_idx) + 
                F.cross_entropy(z, size_idx))
    
    def predict(self, color_idx, shape_idx, size_idx):
        self.eval()
        with torch.no_grad():
            c, s, z = self.forward(color_idx, shape_idx, size_idx)
        self.train()
        return c.argmax(-1), s.argmax(-1), z.argmax(-1)


class FactorizedAE3(nn.Module):
    """Factorized autoencoder for 3-factor domain."""
    
    def __init__(self, n_colors, n_shapes, n_sizes, embed_dim=4, hidden_dim=8, latent_dim=6):
        super().__init__()
        
        self.color_embed = nn.Embedding(n_colors, embed_dim)
        self.shape_embed = nn.Embedding(n_shapes, embed_dim)
        self.size_embed = nn.Embedding(n_sizes, embed_dim)
        
        factor_latent = latent_dim // 3
        
        self.color_encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, factor_latent)
        )
        self.shape_encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, factor_latent)
        )
        self.size_encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, factor_latent)
        )
        
        self.color_decoder = nn.Sequential(
            nn.Linear(factor_latent, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_colors)
        )
        self.shape_decoder = nn.Sequential(
            nn.Linear(factor_latent, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_shapes)
        )
        self.size_decoder = nn.Sequential(
            nn.Linear(factor_latent, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_sizes)
        )
    
    def forward(self, color_idx, shape_idx, size_idx):
        z_c = self.color_encoder(self.color_embed(color_idx))
        z_s = self.shape_encoder(self.shape_embed(shape_idx))
        z_z = self.size_encoder(self.size_embed(size_idx))
        return self.color_decoder(z_c), self.shape_decoder(z_s), self.size_decoder(z_z)
    
    def loss(self, color_idx, shape_idx, size_idx):
        c, s, z = self.forward(color_idx, shape_idx, size_idx)
        return (F.cross_entropy(c, color_idx) + 
                F.cross_entropy(s, shape_idx) + 
                F.cross_entropy(z, size_idx))
    
    def predict(self, color_idx, shape_idx, size_idx):
        self.eval()
        with torch.no_grad():
            c, s, z = self.forward(color_idx, shape_idx, size_idx)
        self.train()
        return c.argmax(-1), s.argmax(-1), z.argmax(-1)


# =============================================================================
# MODEL: PARTIAL FACTORIZATION (Study 3)
# =============================================================================

class PartialFactorizedAE2(nn.Module):
    """
    Autoencoder with parameterized factorization.
    
    share_ratio = 0.0: Fully factorized (separate encoders)
    share_ratio = 1.0: Fully holistic (shared encoder)
    share_ratio = 0.5: Half shared, half separate
    
    Implementation: Latent = [shared_dims, color_dims, shape_dims]
    share_ratio controls proportion allocated to shared vs. separate
    """
    
    def __init__(self, n_colors, n_shapes, embed_dim=4, hidden_dim=8, 
                 latent_dim=4, share_ratio=0.5):
        super().__init__()
        self.n_colors = n_colors
        self.n_shapes = n_shapes
        self.share_ratio = share_ratio
        
        # Calculate dimension allocation
        shared_dim = int(latent_dim * share_ratio)
        factor_dim = (latent_dim - shared_dim) // 2
        
        # Ensure at least 1 dim each if ratio < 1
        if share_ratio < 1.0 and factor_dim == 0:
            factor_dim = 1
            shared_dim = latent_dim - 2 * factor_dim
        
        self.shared_dim = shared_dim
        self.factor_dim = factor_dim
        
        # Embeddings
        self.color_embed = nn.Embedding(n_colors, embed_dim)
        self.shape_embed = nn.Embedding(n_shapes, embed_dim)
        
        # Shared encoder (sees both)
        if shared_dim > 0:
            self.shared_encoder = nn.Sequential(
                nn.Linear(embed_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, shared_dim)
            )
        else:
            self.shared_encoder = None
        
        # Factor-specific encoders
        if factor_dim > 0:
            self.color_encoder = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, factor_dim)
            )
            self.shape_encoder = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, factor_dim)
            )
        else:
            self.color_encoder = None
            self.shape_encoder = None
        
        # Decoders use appropriate latent portions
        total_for_color = shared_dim + factor_dim
        total_for_shape = shared_dim + factor_dim
        
        self.color_decoder = nn.Sequential(
            nn.Linear(total_for_color, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_colors)
        )
        self.shape_decoder = nn.Sequential(
            nn.Linear(total_for_shape, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_shapes)
        )
    
    def forward(self, color_idx, shape_idx):
        e_color = self.color_embed(color_idx)
        e_shape = self.shape_embed(shape_idx)
        
        latent_parts_color = []
        latent_parts_shape = []
        
        # Shared latent (sees both inputs)
        if self.shared_dim > 0 and self.shared_encoder is not None:
            combined = torch.cat([e_color, e_shape], dim=-1)
            z_shared = self.shared_encoder(combined)
            latent_parts_color.append(z_shared)
            latent_parts_shape.append(z_shared)
        
        # Factor-specific latents
        if self.factor_dim > 0 and self.color_encoder is not None:
            z_color = self.color_encoder(e_color)
            z_shape = self.shape_encoder(e_shape)
            latent_parts_color.append(z_color)
            latent_parts_shape.append(z_shape)
        
        # Decode
        z_for_color = torch.cat(latent_parts_color, dim=-1)
        z_for_shape = torch.cat(latent_parts_shape, dim=-1)
        
        return self.color_decoder(z_for_color), self.shape_decoder(z_for_shape)
    
    def loss(self, color_idx, shape_idx):
        color_logits, shape_logits = self.forward(color_idx, shape_idx)
        return (F.cross_entropy(color_logits, color_idx) + 
                F.cross_entropy(shape_logits, shape_idx))
    
    def predict(self, color_idx, shape_idx):
        self.eval()
        with torch.no_grad():
            color_logits, shape_logits = self.forward(color_idx, shape_idx)
        self.train()
        return color_logits.argmax(-1), shape_logits.argmax(-1)


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

def train_model(model, train_tensors, epochs=EPOCHS, lr=LR):
    """Train any model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = model.loss(*train_tensors)
        loss.backward()
        optimizer.step()
    
    return loss.item()


def evaluate_model_2factor(model, compositions, color2idx, shape2idx):
    """Evaluate 2-factor model."""
    colors = torch.tensor([color2idx[c] for c, s in compositions])
    shapes = torch.tensor([shape2idx[s] for c, s in compositions])
    
    pred_c, pred_s = model.predict(colors, shapes)
    correct = ((pred_c == colors) & (pred_s == shapes)).float().mean().item()
    return correct


def evaluate_model_3factor(model, compositions, color2idx, shape2idx, size2idx):
    """Evaluate 3-factor model."""
    colors = torch.tensor([color2idx[c] for c, s, z in compositions])
    shapes = torch.tensor([shape2idx[s] for c, s, z in compositions])
    sizes = torch.tensor([size2idx[z] for c, s, z in compositions])
    
    pred_c, pred_s, pred_z = model.predict(colors, shapes, sizes)
    correct = ((pred_c == colors) & (pred_s == shapes) & (pred_z == sizes)).float().mean().item()
    return correct


# =============================================================================
# STUDY 1: HOLDOUT SWEEP
# =============================================================================

def run_study1():
    """
    Study 1: Vary number of held-out compositions.
    
    Question: Does the factorization advantage persist as we demand
    more generalization (fewer training examples, more test)?
    """
    print("\n" + "=" * 70)
    print("STUDY 1: HOLDOUT SWEEP")
    print("=" * 70)
    print("Domain: 3 colors × 3 shapes = 9 compositions")
    print("Varying holdout: 2, 3, 4, 5 compositions")
    
    colors = COLORS_3
    shapes = SHAPES_3
    color2idx = {c: i for i, c in enumerate(colors)}
    shape2idx = {s: i for i, s in enumerate(shapes)}
    
    all_compositions = [(c, s) for c in colors for s in shapes]
    
    results = {"holdout_sizes": [], "holistic_test": [], "factorized_test": []}
    
    print(f"\n{'Holdout':<10} {'Train':<8} {'Holistic Test':<18} {'Factorized Test':<18}")
    print("-" * 60)
    
    for n_holdout in [2, 3, 4, 5]:
        holistic_accs = []
        factorized_accs = []
        
        for run in range(N_RUNS):
            set_seed(SEED + run)
            
            train, test = create_compositional_holdout(
                all_compositions, n_holdout, 
                n_factors=2, 
                seed=SEED + run + n_holdout * 100
            )
            
            train_colors = torch.tensor([color2idx[c] for c, s in train])
            train_shapes = torch.tensor([shape2idx[s] for c, s in train])
            
            # Holistic
            model_h = HolisticAE2(len(colors), len(shapes))
            train_model(model_h, (train_colors, train_shapes))
            acc_h = evaluate_model_2factor(model_h, test, color2idx, shape2idx)
            holistic_accs.append(acc_h)
            
            # Factorized
            model_f = FactorizedAE2(len(colors), len(shapes))
            train_model(model_f, (train_colors, train_shapes))
            acc_f = evaluate_model_2factor(model_f, test, color2idx, shape2idx)
            factorized_accs.append(acc_f)
        
        h_mean, h_std = np.mean(holistic_accs), np.std(holistic_accs)
        f_mean, f_std = np.mean(factorized_accs), np.std(factorized_accs)
        
        print(f"{n_holdout:<10} {9-n_holdout:<8} {h_mean*100:>5.1f}% ± {h_std*100:>4.1f}%    "
              f"{f_mean*100:>5.1f}% ± {f_std*100:>4.1f}%")
        
        results["holdout_sizes"].append(n_holdout)
        results["holistic_test"].append({"mean": h_mean, "std": h_std})
        results["factorized_test"].append({"mean": f_mean, "std": f_std})
    
    print("\nInterpretation:")
    print("  If factorized consistently outperforms holistic across holdout sizes,")
    print("  the effect is robust to increased generalization demand.")
    
    return results


# =============================================================================
# STUDY 2: HARDER DOMAIN
# =============================================================================

def run_study2():
    """
    Study 2: 3-factor domain (color × shape × size).
    
    Question: Does factorization scale to more factors?
    """
    print("\n" + "=" * 70)
    print("STUDY 2: HARDER DOMAIN (3 FACTORS)")
    print("=" * 70)
    print("Domain: 4 colors × 4 shapes × 3 sizes = 48 compositions")
    print("Holdout: 12 compositions (25%)")
    
    colors = COLORS_4
    shapes = SHAPES_4
    sizes = SIZES_3
    
    color2idx = {c: i for i, c in enumerate(colors)}
    shape2idx = {s: i for i, s in enumerate(shapes)}
    size2idx = {z: i for i, z in enumerate(sizes)}
    
    all_compositions = [(c, s, z) for c in colors for s in shapes for z in sizes]
    
    holistic_accs = []
    factorized_accs = []
    
    print(f"\n{'Run':<6} {'Holistic':<12} {'Factorized':<12}")
    print("-" * 35)
    
    for run in range(N_RUNS):
        set_seed(SEED + run)
        
        train, test = create_compositional_holdout(
            all_compositions, 12, 
            n_factors=3,
            seed=SEED + run
        )
        
        train_c = torch.tensor([color2idx[c] for c, s, z in train])
        train_s = torch.tensor([shape2idx[s] for c, s, z in train])
        train_z = torch.tensor([size2idx[z] for c, s, z in train])
        
        # Holistic
        model_h = HolisticAE3(len(colors), len(shapes), len(sizes))
        train_model(model_h, (train_c, train_s, train_z))
        acc_h = evaluate_model_3factor(model_h, test, color2idx, shape2idx, size2idx)
        holistic_accs.append(acc_h)
        
        # Factorized
        model_f = FactorizedAE3(len(colors), len(shapes), len(sizes))
        train_model(model_f, (train_c, train_s, train_z))
        acc_f = evaluate_model_3factor(model_f, test, color2idx, shape2idx, size2idx)
        factorized_accs.append(acc_f)
        
        print(f"{run+1:<6} {acc_h*100:>6.1f}%      {acc_f*100:>6.1f}%")
    
    h_mean, h_std = np.mean(holistic_accs), np.std(holistic_accs)
    f_mean, f_std = np.mean(factorized_accs), np.std(factorized_accs)
    
    print("-" * 35)
    print(f"{'Mean':<6} {h_mean*100:>5.1f}% ± {h_std*100:.1f}%  {f_mean*100:>5.1f}% ± {f_std*100:.1f}%")
    
    diff = f_mean - h_mean
    print(f"\nDifference: {diff*100:+.1f}%")
    print("\nInterpretation:")
    if diff > 0.1:
        print("  Factorization advantage scales to 3-factor domain.")
    elif diff > 0:
        print("  Modest factorization advantage in 3-factor domain.")
    else:
        print("  No factorization advantage — may need investigation.")
    
    return {
        "holistic": {"mean": h_mean, "std": h_std, "all": holistic_accs},
        "factorized": {"mean": f_mean, "std": f_std, "all": factorized_accs}
    }


# =============================================================================
# STUDY 3: PARTIAL FACTORIZATION
# =============================================================================

def run_study3():
    """
    Study 3: Gradient of factorization.
    
    Question: Is compositionality binary (factorized vs. not) or graded?
    """
    print("\n" + "=" * 70)
    print("STUDY 3: PARTIAL FACTORIZATION")
    print("=" * 70)
    print("Domain: 3 colors × 3 shapes = 9 compositions")
    print("Holdout: 2 compositions")
    print("Varying share_ratio: 0.0 (fully factorized) to 1.0 (fully holistic)")
    
    colors = COLORS_3
    shapes = SHAPES_3
    color2idx = {c: i for i, c in enumerate(colors)}
    shape2idx = {s: i for i, s in enumerate(shapes)}
    
    all_compositions = [(c, s) for c in colors for s in shapes]
    
    share_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
    results = {"share_ratios": share_ratios, "test_accs": []}
    
    print(f"\n{'Share Ratio':<15} {'Description':<20} {'Test Accuracy':<18}")
    print("-" * 55)
    
    for share_ratio in share_ratios:
        accs = []
        
        for run in range(N_RUNS):
            set_seed(SEED + run)
            
            train, test = create_compositional_holdout(
                all_compositions, 2,
                n_factors=2,
                seed=SEED + run
            )
            
            train_c = torch.tensor([color2idx[c] for c, s in train])
            train_s = torch.tensor([shape2idx[s] for c, s in train])
            
            model = PartialFactorizedAE2(
                len(colors), len(shapes),
                latent_dim=4,
                share_ratio=share_ratio
            )
            train_model(model, (train_c, train_s))
            acc = evaluate_model_2factor(model, test, color2idx, shape2idx)
            accs.append(acc)
        
        mean_acc, std_acc = np.mean(accs), np.std(accs)
        
        if share_ratio == 0.0:
            desc = "Fully factorized"
        elif share_ratio == 1.0:
            desc = "Fully holistic"
        else:
            desc = f"{int(share_ratio*100)}% shared"
        
        print(f"{share_ratio:<15.2f} {desc:<20} {mean_acc*100:>5.1f}% ± {std_acc*100:.1f}%")
        
        results["test_accs"].append({"mean": mean_acc, "std": std_acc})
    
    # Analyze gradient
    print("\nInterpretation:")
    accs_by_share = [r["mean"] for r in results["test_accs"]]
    
    if accs_by_share[0] > accs_by_share[-1] + 0.1:
        # Check if gradient is smooth
        is_monotonic = all(accs_by_share[i] >= accs_by_share[i+1] - 0.05 
                          for i in range(len(accs_by_share)-1))
        if is_monotonic:
            print("  Graded effect: More factorization → better composition (smooth gradient)")
        else:
            print("  Non-monotonic: Relationship is complex")
    else:
        print("  No clear gradient detected")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("EXTENDED COMPOSITIONAL ABSTRACTION EXPERIMENTS")
    print("=" * 70)
    print(f"Runs per condition: {N_RUNS}")
    print(f"Epochs per model: {EPOCHS}")
    print(f"Base seed: {SEED}")
    
    all_results = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "n_runs": N_RUNS,
            "epochs": EPOCHS,
            "seed": SEED
        }
    }
    
    # Run studies
    all_results["study1_holdout_sweep"] = run_study1()
    all_results["study2_harder_domain"] = run_study2()
    all_results["study3_partial_factorization"] = run_study3()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print("""
Study 1 (Holdout Sweep):
  Tests robustness to increased generalization demand.
  If factorized wins across all holdout sizes → effect is robust.

Study 2 (3-Factor Domain):
  Tests scaling to more factors.
  If factorized wins → architectural factorization scales.

Study 3 (Partial Factorization):
  Tests whether compositionality is binary or graded.
  If smooth gradient → can tune factorization/compression tradeoff.
""")
    
    # Save
    output_path = Path("results/extended_experiments.json")
    output_path.parent.mkdir(exist_ok=True)
    
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=convert)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()