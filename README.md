# Compositional Abstraction

A minimal experiment testing whether factorization pressure improves compositional generalization.

## Theoretical Background

**Core Question**: Does architectural pressure toward factorized representations enable generalization to novel compositions?

**Framework**: Abstraction Primitive Hypothesis (APH)
- Hypothesis: Abstraction (factorized, composable representations) differs from compression (holistic encoding)
- Prediction: Systems with factorization pressure should generalize better to novel compositions

**Established Literature**:
- β-VAE encourages disentangled representations (Higgins et al., 2017, ICLR)
- Transformers struggle with compositional generalization (Lake & Baroni, 2018, ICML)
- Disentanglement requires inductive biases (Locatello et al., 2019, ICML)

## Design

**Domain**: Two-word phrases (color + shape)
- Colors: red, blue, green
- Shapes: circle, square, triangle
- 9 total compositions

**Train/Test Split**: 
- Training: 7 compositions
- Test: 2 held-out NOVEL compositions (factors seen, combination not)

**Models**:
- Standard VAE (β=1): Compression baseline
- β-VAE (β=4): Factorization pressure

**Prediction**: β-VAE shows smaller generalization gap (higher test accuracy)

## Usage
```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run experiment
python3 src/experiment.py
```

## Results

Results saved to `results/experiment_results.json`

## Author

Hillary Danan, PhD  
Cognitive Neuroscience

## License

MIT