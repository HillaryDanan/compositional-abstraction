"""
Compositional Abstraction Experiment (v3)
==========================================

Update: Test ARCHITECTURAL factorization, not just loss-based

Key change:
    - Holistic model: single encoder sees both words
    - Factorized model: separate encoders for color and shape
    
This directly tests whether factorized representations enable composition,
rather than hoping β-VAE discovers factorization.

Theoretical note:
    This aligns with Andreas et al. (2016) Neural Module Networks -
    architectural modularity as inductive bias for compositionality.

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

COLORS = ["red", "blue", "green"]
SHAPES = ["circle", "square", "triangle"]

# Separate vocabularies for factorized model
COLOR2IDX = {c: i for i, c in enumerate(COLORS)}
SHAPE2IDX = {s: i for i, s in enumerate(SHAPES)}

# Combined vocabulary for holistic model
VOCAB = COLORS + SHAPES
WORD2IDX = {w: i for i, w in enumerate(VOCAB)}

ALL_COMPOSITIONS = [(c, s) for c in COLORS for s in SHAPES]

# Hyperparameters
EMBED_DIM = 4        # Small embeddings
LATENT_DIM = 4       # 2 dims per factor
HIDDEN_DIM = 8       # Minimal
EPOCHS = 1000        
LR = 0.01
SEED = 42
N_RUNS = 20          # More runs for reliability


# =============================================================================
# MODEL A: HOLISTIC (baseline)
# =============================================================================

class HolisticAutoencoder(nn.Module):
    """
    Standard autoencoder: encodes both words together.
    
    This is the "compression" baseline - no structural pressure
    to separate color from shape.
    """
    
    def __init__(self):
        super().__init__()
        
        # Single embedding for all words
        self.embed = nn.Embedding(len(VOCAB), EMBED_DIM)
        
        # Encoder: concat both embeddings → latent
        self.encoder = nn.Sequential(
            nn.Linear(EMBED_DIM * 2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, LATENT_DIM)
        )
        
        # Decoder: latent → both predictions
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN_DIM),
            nn.ReLU()
        )
        self.color_head = nn.Linear(HIDDEN_DIM, len(COLORS))
        self.shape_head = nn.Linear(HIDDEN_DIM, len(SHAPES))
    
    def forward(self, color_idx, shape_idx):
        # Encode holistically
        e_color = self.embed(color_idx)
        e_shape = self.embed(shape_idx)
        x = torch.cat([e_color, e_shape], dim=-1)
        z = self.encoder(x)
        
        # Decode
        h = self.decoder(z)
        color_logits = self.color_head(h)
        shape_logits = self.shape_head(h)
        
        return color_logits, shape_logits, z
    
    def loss(self, color_idx, shape_idx):
        color_logits, shape_logits, _ = self.forward(color_idx, shape_idx)
        loss = (F.cross_entropy(color_logits, color_idx) + 
                F.cross_entropy(shape_logits, shape_idx))
        return loss
    
    def predict(self, color_idx, shape_idx):
        self.eval()
        with torch.no_grad():
            color_logits, shape_logits, _ = self.forward(color_idx, shape_idx)
            pred_color = color_logits.argmax(dim=-1)
            pred_shape = shape_logits.argmax(dim=-1)
        self.train()
        return pred_color, pred_shape


# =============================================================================
# MODEL B: FACTORIZED (architectural compositionality)
# =============================================================================

class FactorizedAutoencoder(nn.Module):
    """
    Factorized autoencoder: SEPARATE encoders for color and shape.
    
    Key insight: This model CANNOT encode "red circle" as a holistic unit.
    It must encode color and shape through independent pathways.
    
    The latent space is explicitly [z_color, z_shape].
    
    This is the "abstraction" model - forced compositional structure.
    """
    
    def __init__(self):
        super().__init__()
        
        # SEPARATE embeddings for colors and shapes
        self.color_embed = nn.Embedding(len(COLORS), EMBED_DIM)
        self.shape_embed = nn.Embedding(len(SHAPES), EMBED_DIM)
        
        # SEPARATE encoders - each only sees its own input
        self.color_encoder = nn.Sequential(
            nn.Linear(EMBED_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, LATENT_DIM // 2)  # Half the latent dims
        )
        self.shape_encoder = nn.Sequential(
            nn.Linear(EMBED_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, LATENT_DIM // 2)  # Other half
        )
        
        # SEPARATE decoders - each only uses its own latent dims
        self.color_decoder = nn.Sequential(
            nn.Linear(LATENT_DIM // 2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, len(COLORS))
        )
        self.shape_decoder = nn.Sequential(
            nn.Linear(LATENT_DIM // 2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, len(SHAPES))
        )
    
    def forward(self, color_idx, shape_idx):
        # Encode SEPARATELY
        e_color = self.color_embed(color_idx)
        e_shape = self.shape_embed(shape_idx)
        
        z_color = self.color_encoder(e_color)
        z_shape = self.shape_encoder(e_shape)
        
        # Full latent is concatenation (for analysis)
        z = torch.cat([z_color, z_shape], dim=-1)
        
        # Decode SEPARATELY
        color_logits = self.color_decoder(z_color)
        shape_logits = self.shape_decoder(z_shape)
        
        return color_logits, shape_logits, z
    
    def loss(self, color_idx, shape_idx):
        color_logits, shape_logits, _ = self.forward(color_idx, shape_idx)
        loss = (F.cross_entropy(color_logits, color_idx) + 
                F.cross_entropy(shape_logits, shape_idx))
        return loss
    
    def predict(self, color_idx, shape_idx):
        self.eval()
        with torch.no_grad():
            color_logits, shape_logits, _ = self.forward(color_idx, shape_idx)
            pred_color = color_logits.argmax(dim=-1)
            pred_shape = shape_logits.argmax(dim=-1)
        self.train()
        return pred_color, pred_shape


# =============================================================================
# DATA
# =============================================================================

def create_train_test_split(holdout_compositions, seed=SEED):
    """Create compositional train/test split."""
    random.seed(seed)
    
    test = holdout_compositions
    train = [c for c in ALL_COMPOSITIONS if c not in test]
    
    # Verify compositional structure
    train_colors = set(c[0] for c in train)
    train_shapes = set(c[1] for c in train)
    
    for color, shape in test:
        assert color in train_colors, f"Color '{color}' not in training"
        assert shape in train_shapes, f"Shape '{shape}' not in training"
    
    return train, test


def compositions_to_tensors(compositions):
    """Convert to tensors using separate indexing."""
    colors = torch.tensor([COLOR2IDX[c] for c, s in compositions])
    shapes = torch.tensor([SHAPE2IDX[s] for c, s in compositions])
    return colors, shapes


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

def train_model(model, train_compositions, epochs=EPOCHS, lr=LR):
    """Train model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    colors, shapes = compositions_to_tensors(train_compositions)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = model.loss(colors, shapes)
        loss.backward()
        optimizer.step()
    
    return loss.item()


def evaluate_model(model, compositions):
    """Evaluate accuracy."""
    colors, shapes = compositions_to_tensors(compositions)
    pred_colors, pred_shapes = model.predict(colors, shapes)
    
    correct_colors = (pred_colors == colors).float()
    correct_shapes = (pred_shapes == shapes).float()
    correct_phrases = (correct_colors * correct_shapes).mean().item()
    
    return {
        "accuracy": correct_phrases,
        "color_accuracy": correct_colors.mean().item(),
        "shape_accuracy": correct_shapes.mean().item()
    }


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment(holdout_compositions, n_runs=N_RUNS, seed_base=SEED):
    """
    Compare holistic vs. factorized architectures.
    
    Prediction (from APH):
        - Both should achieve 100% training accuracy
        - Factorized should show higher test accuracy (compositional generalization)
        - Because factorized representations MUST compose - there's no other option
    """
    train_compositions, test_compositions = create_train_test_split(
        holdout_compositions, seed=seed_base
    )
    
    print("=" * 70)
    print("COMPOSITIONAL ABSTRACTION EXPERIMENT (v3)")
    print("HOLISTIC vs. FACTORIZED ARCHITECTURE")
    print("=" * 70)
    
    print(f"\nTraining compositions ({len(train_compositions)}):")
    for c, s in train_compositions:
        print(f"  {c} {s}")
    
    print(f"\nHeld-out compositions ({len(test_compositions)}):")
    for c, s in test_compositions:
        print(f"  {c} {s}")
    
    print(f"\nRuns: {n_runs}")
    
    results = {
        "holistic": {"train": [], "test": []},
        "factorized": {"train": [], "test": []}
    }
    
    # Run experiments
    for model_type in ["holistic", "factorized"]:
        print(f"\n{'='*70}")
        print(f"MODEL: {model_type.upper()}")
        print("=" * 70)
        
        for run in range(n_runs):
            torch.manual_seed(seed_base + run)
            np.random.seed(seed_base + run)
            
            if model_type == "holistic":
                model = HolisticAutoencoder()
            else:
                model = FactorizedAutoencoder()
            
            final_loss = train_model(model, train_compositions)
            train_eval = evaluate_model(model, train_compositions)
            test_eval = evaluate_model(model, test_compositions)
            
            results[model_type]["train"].append(train_eval["accuracy"])
            results[model_type]["test"].append(test_eval["accuracy"])
            
            if (run + 1) % 5 == 0:
                print(f"  Run {run+1}: train={train_eval['accuracy']*100:.0f}%, "
                      f"test={test_eval['accuracy']*100:.0f}%")
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\n{'Model':<15} {'Train Acc':<20} {'Test Acc':<20} {'Gen. Gap':<15}")
    print("-" * 70)
    
    for model_type in ["holistic", "factorized"]:
        train_mean = np.mean(results[model_type]["train"])
        train_std = np.std(results[model_type]["train"])
        test_mean = np.mean(results[model_type]["test"])
        test_std = np.std(results[model_type]["test"])
        gap = train_mean - test_mean
        
        print(f"{model_type:<15} {train_mean*100:>5.1f}% ± {train_std*100:>4.1f}%    "
              f"{test_mean*100:>5.1f}% ± {test_std*100:>4.1f}%    {gap*100:>5.1f}%")
        
        results[model_type]["summary"] = {
            "train_mean": train_mean,
            "train_std": train_std,
            "test_mean": test_mean,
            "test_std": test_std,
            "gap": gap
        }
    
    # Statistical comparison
    print("\n" + "-" * 70)
    print("STATISTICAL COMPARISON")
    print("-" * 70)
    
    holistic_test = results["holistic"]["test"]
    factorized_test = results["factorized"]["test"]
    
    diff = np.mean(factorized_test) - np.mean(holistic_test)
    
    # Simple t-test (scipy may not be needed for this)
    pooled_std = np.sqrt((np.var(holistic_test) + np.var(factorized_test)) / 2)
    if pooled_std > 0:
        t_stat = diff / (pooled_std * np.sqrt(2 / n_runs))
    else:
        t_stat = float('inf') if diff > 0 else float('-inf') if diff < 0 else 0
    
    print(f"\nFactorized - Holistic test accuracy: {diff*100:+.1f}%")
    print(f"Effect size (Cohen's d): {diff / pooled_std if pooled_std > 0 else 'undefined':.2f}")
    
    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    if diff > 0.1:
        print(f"""
Factorized architecture shows {diff*100:.1f}% better compositional generalization.

→ SUPPORTS APH: Architectural factorization enables composition.
  The factorized model MUST encode color and shape separately.
  When tested on novel combinations, it can compose because
  the representations were never entangled.

→ This is not about β-VAE discovering disentanglement.
  This is about FORCING compositional structure architecturally.
""")
    elif diff > 0:
        print(f"""
Factorized architecture shows modest improvement ({diff*100:.1f}%).

→ WEAK SUPPORT: Some benefit from architectural factorization.
→ Effect may be small due to domain simplicity.
""")
    elif diff < -0.1:
        print(f"""
Holistic architecture shows better generalization ({-diff*100:.1f}%).

→ CHALLENGES APH: Factorization may not help in this domain.
→ Requires investigation: Is the task actually compositional?
""")
    else:
        print(f"""
No meaningful difference between architectures ({diff*100:.1f}%).

→ INCONCLUSIVE: Domain may be too simple to discriminate.
→ Or: Both architectures solving via same mechanism.
""")
    
    return results


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    HOLDOUT = [("blue", "triangle"), ("green", "circle")]
    
    results = run_experiment(HOLDOUT, n_runs=20)
    
    # Save
    output_path = Path("results/architecture_comparison.json")
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    
    print(f"\nResults saved to: {output_path}")