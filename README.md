# i_rewrote_pytorch

A neural-network library built from scratch in pure NumPy — a "why buy when you can build" reimplementation of the core ideas behind PyTorch/TensorFlow, done for fun and learning.

> This is a procrastination project I started one day when I was supposed to be doing homework. The plan is to keep bolting on as many different machine-learning pieces as I can. There's no real reason to use this over PyTorch or TensorFlow — I'm not optimizing for hardware — but the math isn't bad, and building it yourself is the best way to understand it.

## What it does

- **Feed-forward neural networks** of arbitrary depth and width (`neural_network.py`), with configurable per-layer activations (sigmoid, ReLU, linear).
- **Training from scratch** — forward propagation, backpropagation, and gradient descent, all implemented by hand in NumPy.
- **Weight persistence** — save and reload trained parameters as `.npz` files.
- **Worked example** — training a network to classify handwritten digits on the MNIST dataset (`number_id_training_ex.py`).

## Repository layout

| File | Purpose |
|------|---------|
| `neural_network.py` | The `NeuralNetwork` class — layers, activations, forward/back propagation, training |
| `theta_init.py` | Weight-initialization strategies (normal, logistic) |
| `number_id_training_ex.py` | Example: train and evaluate an MNIST digit classifier |
| `test.py` | Scratch / testing script |
| `test_library/MNIST_CSV/` | MNIST data in CSV form plus a helper to (re)generate it |

## Requirements

- Python 3.9+
- `numpy`, `pandas`, `matplotlib`

```bash
pip install numpy pandas matplotlib
```

## Running the MNIST example

The MNIST data is stored as CSV (`label, pix-11, pix-12, ...`) under `test_library/MNIST_CSV/`; see [MNIST in CSV](https://pjreddie.com/projects/mnist-in-csv/) for the format.

```bash
python number_id_training_ex.py
```

## Notes

Educational project — not intended for production use. Contributions of new ML "things" to the library are the whole point.
