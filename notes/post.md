# Compositional Generalization Requires Architectural Factorization: A Minimal Demonstration

## Summary

I tested whether architectural factorization enables compositional generalization. Results:

- Holistic architecture: 100% train accuracy, 2.5% test accuracy on novel compositions
- Factorized architecture: 100% train accuracy, 100% test accuracy on novel compositions
- Effect size: d = 12.65
- Effect is robust across holdout sizes and scales to 3-factor domains
- Partial factorization yields unstable, not partial, composition

This supports the claim that factorized representations enable composition, while holistic compression enables only memorization.

## Background

Compositional generalization — the ability to understand novel combinations of known components — is a key challenge for neural networks. Lake & Baroni (2018) demonstrated that sequence-to-sequence models fail systematically on compositional tasks, even when they achieve perfect training accuracy.

The standard explanation is that neural networks learn holistic, entangled representations that don't support recombination. But this raises a question: if we force factorized representations architecturally, does compositional generalization follow?

## Theoretical Framework

This experiment tests a prediction from the Abstraction Primitive Hypothesis (APH): that factorized representations enable composition, while holistic representations do not.

The key distinction:
- **Holistic encoding**: "red circle" is stored as a single pattern
- **Factorized encoding**: "red" and "circle" are stored separately and combined at output

If APH is correct, only factorized systems should generalize to novel compositions like "blue triangle" after training on "blue square" and "red triangle."

## Experiment 1: Holistic vs. Factorized

### Domain

Two-word phrases: 3 colors × 3 shapes = 9 compositions

- Training: 7 compositions
- Test: 2 held-out compositions where both factors were seen but never together

### Models

**Holistic Autoencoder**:
- Single embedding layer for all words
- Encoder sees concatenated [color, shape] embeddings
- Can encode "red circle" as a holistic unit

**Factorized Autoencoder**:
- Separate embedding layers for colors and shapes
- Separate encoders that only see their own input
- Cannot encode "red circle" holistically — must encode color and shape independently

Both models have identical capacity (~500 parameters). The only difference is architectural.

### Results

| Model | Train Accuracy | Test Accuracy |
|-------|----------------|---------------|
| Holistic | 100.0% ± 0.0% | 2.5% ± 10.9% |
| Factorized | 100.0% ± 0.0% | 100.0% ± 0.0% |

Effect size: d = 12.65

The holistic model achieves perfect training accuracy but cannot generalize to novel compositions. The factorized model achieves perfect generalization.

## Experiment 2: Robustness Tests

### Study 1: Holdout Sweep

Does the effect hold as we demand more generalization?

| Holdout | Holistic Test | Factorized Test |
|---------|---------------|-----------------|
| 2 of 9 | 5.0% | 100.0% |
| 3 of 9 | 6.7% | 86.7% |
| 4 of 9 | 2.5% | 100.0% |
| 5 of 9 | 6.0% | 100.0% |

The effect is robust. Even training on only 4 compositions, the factorized model achieves 100% test accuracy.

### Study 2: Scaling to 3 Factors

Does factorization scale to more complex domains?

Domain: 4 colors × 4 shapes × 3 sizes = 48 compositions
Holdout: 12 (25%)

| Model | Test Accuracy |
|-------|---------------|
| Holistic | 64.2% ± 17.5% |
| Factorized | 100.0% ± 0.0% |

The factorized model achieves perfect generalization with zero variance. The holistic model is unreliable (33-92% across runs).

### Study 3: Partial Factorization

Is compositionality binary or graded?

| Share Ratio | Test Accuracy |
|-------------|---------------|
| 0% (fully factorized) | 100.0% ± 0.0% |
| 25% shared | 45.0% ± 26.9% |
| 50% shared | 70.0% ± 33.2% |
| 100% (fully holistic) | 15.0% ± 22.9% |

The relationship is non-monotonic. Key finding: partial factorization doesn't yield partial composition — it yields unstable composition. Once the architecture can encode holistically, it sometimes will, destroying reliable generalization.

This suggests compositionality may be closer to binary than graded.

## Interpretation

### What This Shows

1. **Factorization enables composition**: When representations must be factorized, composition follows automatically.
2. **The effect is robust**: Holds across holdout sizes and scales to more factors.
3. **Compositionality may be binary**: Partial factorization yields instability, not graceful degradation.

### What This Does Not Show

1. **That systems can discover what to factorize**: We forced factorization. The hard problem — learning what should be separate — remains open.
2. **That this scales to realistic domains**: We tested minimal synthetic domains.
3. **That "abstraction" is the fundamental primitive of intelligence**: That's the larger theoretical claim. This demonstrates one mechanism consistent with that claim.

### Relation to LLMs

Current LLMs are holistic encoders. They can memorize vast amounts of compositional structure from training data, achieving apparent compositional ability. But if this analysis is correct, they are not genuinely composing — they are pattern-matching over memorized compositions.

This predicts:
- LLMs should fail on truly novel compositions (not seen in training)
- Scaling parameters should not fundamentally solve this (you can memorize more, but not compose)
- Architectural changes (not just scale) are needed for genuine composition

## Limitations

- Minimal synthetic domains
- Small models (~500 parameters)
- We forced factorization rather than testing whether it can be learned
- Binary classification of "correct" vs "incorrect" may miss partial understanding

## Code

Available at: https://github.com/HillaryDanan/compositional-abstraction

## References

- Lake, B. & Baroni, M. (2018). Generalization without systematicity: On the compositional skills of sequence-to-sequence recurrent networks. ICML.
- Andreas, J., Rohrbach, M., Darrell, T., & Klein, D. (2016). Neural module networks. CVPR.
- Locatello, F., et al. (2019). Challenging common assumptions in the unsupervised learning of disentangled representations. ICML.
- Higgins, I., et al. (2017). β-VAE: Learning basic visual concepts with a constrained variational framework. ICLR.

## Acknowledgments

This work tests predictions from the Abstraction Primitive Hypothesis framework. See: https://github.com/HillaryDanan/abstraction-intelligence