Project: MNIST Classifiers

This repository implements several simple classifiers for MNIST provided as raw PNG images in the `MNIST/` folder. Implementations include:
- K-Nearest Neighbors (NumPy)
- Bernoulli Naive Bayes (NumPy)
- Linear classifier (L2 loss) (PyTorch)
- Multilayer Perceptron (PyTorch)
- Convolutional Neural Network (PyTorch)

Quick environment setup (CPU)
```bash
python -m pip install --upgrade pip
python -m pip install numpy pillow matplotlib scikit-learn tqdm torch
```
If you have CUDA, install the matching CUDA-enabled PyTorch from https://pytorch.org/get-started/locally/.

How to run
- RUN ALL:
  ```bash
  bash run_all.sh
  ```
- MLP (PyTorch):
  ```bash
  python multilayer_perception.py
  ```
- CNN (PyTorch):
  ```bash
  python convolutional_network.py
  ```
- Linear classifier (PyTorch):
  ```bash
  python linear_classifier.py
  ```
- Naive Bayes (NumPy):
  ```bash
  python naive_bayes.py
  ```
- KNN (NumPy): 
  ```bash
  python k_nearest_neighbors.py
  ```

What the scripts do
- Each training script saves results under `results/<model>/<timestamp>/`
  - Classification report (text)
  - Confusion matrix (CSV)
  - Optional visualizations (PNG)

Notes
- The repository uses `preprocessing.py` helpers to load PNG images and return either NumPy arrays (`collect_np_dataset`) or PyTorch tensors (`collect_torch_dataset`).
- Labels are integer class indices (0–9). PyTorch models expect targets as `torch.long` when using `CrossEntropyLoss`.
- `tqdm` is only needed to run `k_nearest_neighbors.py` as it helps determine the length of time to finish processing.

