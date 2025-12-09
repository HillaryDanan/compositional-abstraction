# Compositional Abstraction

A minimal experimental test of whether architectural factorization enables compositional generalization.

## Theoretical Background

**Core Question**: Does architectural pressure toward factorized representations enable generalization to novel compositions?

**Framework**: Abstraction Primitive Hypothesis (APH)
- Hypothesis: Abstraction (factorized, composable representations) differs from compression (holistic encoding)
- Prediction: Systems with factorized representations should generalize to novel compositions; holistic systems should not

**Established Literature**:
- Transformers struggle with compositional generalization (Lake & Baroni, 2018, ICML)
- Architectural modularity improves systematicity (Andreas et al., 2016, Neural Module Networks)
- Disentanglement requires inductive biases (Locatello et al., 2019, ICML)

## Experiments

### Experiment 1: Holistic vs. Factorized Architecture

**Domain**: Two-word phrases (color + shape)
- Colors: red, blue, green
- Shapes: circle, square, triangle
- 9 total compositions

**Design**:
- Training: 7 compositions
- Test: 2 held-out NOVEL compositions (factors seen, combination not)
- Holistic model: Single encoder sees both words
- Factorized model: Separate encoders for color and shape

**Results**:

| Model | Train Accuracy | Test Accuracy | Effect Size |
|-------|----------------|---------------|-------------|
| Holistic | 100% | 2.5% | — |
| Factorized | 100% | 100% | d = 12.65 |

**Interpretation**: Holistic architecture memorizes but cannot compose. Factorized architecture composes perfectly.

### Experiment 2: Extended Studies

#### Study 1: Holdout Sweep

Does the effect hold as we increase generalization demand?

| Holdout | Train Size | Holistic Test | Factorized Test |
|---------|------------|---------------|-----------------|
| 2 | 7 | 5.0% | 100.0% |
| 3 | 6 | 6.7% | 86.7% |
| 4 | 5 | 2.5% | 100.0% |
| 5 | 4 | 6.0% | 100.0% |

**Result**: Effect is robust across holdout sizes.

#### Study 2: Harder Domain (3 Factors)

Does factorization scale to more factors?

Domain: 4 colors x 4 shapes x 3 sizes = 48 compositions
Holdout: 12 compositions (25%)

| Model | Test Accuracy |
|-------|---------------|
| Holistic | 64.2% ± 17.5% |
| Factorized | 100.0% ± 0.0% |

**Result**: Factorization advantage scales to 3-factor domain.

#### Study 3: Partial Factorization

Is compositionality binary or graded?

| Share Ratio | Description | Test Accuracy |
|-------------|-------------|---------------|
| 0.00 | Fully factorized | 100.0% ± 0.0% |
| 0.25 | 25% shared | 45.0% ± 26.9% |
| 0.50 | 50% shared | 70.0% ± 33.2% |
| 0.75 | 75% shared | 70.0% ± 33.2% |
| 1.00 | Fully holistic | 15.0% ± 22.9% |

**Result**: Non-monotonic. Compositionality appears binary rather than graded — partial factorization yields unstable, not partial, composition.

## Summary of Findings

**Strongly supported**:
1. Architectural factorization enables compositional generalization (d > 10)
2. Effect is robust to holdout size
3. Effect scales to more factors
4. Holistic architectures memorize; factorized architectures compose
5. Compositionality may be binary (phase transition) rather than graded — partial factorization yields unstable, not partial, composition

**Not yet tested**:
- Whether systems can discover what to factorize (we forced it)
- Self-reference capabilities
- Scaling to realistic domains

## Usage
```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run main experiment
python3 src/experiment.py

# Run extended experiments
python3 src/extended_experiments.py
```

## Files

- `src/experiment.py` — Main holistic vs. factorized comparison
- `src/extended_experiments.py` — Holdout sweep, 3-factor domain, partial factorization
- `results/` — JSON output from experiments

## Theoretical Context

This work tests predictions from the Abstraction Primitive Hypothesis (APH), which proposes that abstraction — the formation of factorized, composable representations — is the fundamental operation underlying intelligence.

See: [Abstraction-Intelligence](https://github.com/HillaryDanan/Abstraction-Intelligence)

## Author

Hillary Danan, PhD
Cognitive Neuroscience

## License

MIT
