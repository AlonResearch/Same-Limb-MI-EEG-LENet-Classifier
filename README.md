# MI-EEG Motor Imagery Classification

Motor-imagery EEG trials from the MI3 dataset are classified with lightweight LENet variants. Two models are available:

- **LENet (Conv-only classifier block)** – matches the MI2/BCI IV-2a architecture in the original publication.
- **LENet_FCL** – keeps the convolutional front-end but swaps the final block for a dynamically sized fully connected layer to support higher-channel MI3 recordings.

Training, evaluation, and visualization live in `MI3_CNN.ipynb`. Saved weights for the most recent training run are stored as `lenet_ccb_mi3.pth` and `lenet_fcl_mi3.pth`.

## Environment

- Python 3.11 (managed with [uv](https://docs.astral.sh/uv/))
- PyTorch 2.5.1+cu124 (CUDA 12.4), torchvision 0.20+, torchaudio 2.5+
- NumPy 2.1, pandas 2.2, SciPy 1.13, scikit-learn 1.5, matplotlib 3.9

All dependencies are pinned in `pyproject.toml` / `uv.lock` for reproducibility.

## Setup

1. **Install uv (once):** `pip install uv`
2. **Sync the environment:**
	```bash
	uv sync
	```
3. **Activate the virtualenv (optional):** `uv venv && uv pip install -r requirements.txt` is not required; you can run commands via `uv run` directly.
4. **Verify CUDA:**
	```bash
	uv run python - <<'PY'
	import torch
	print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
	PY
	```

## Dataset layout

The repository expects the MI3 BIDS derivative files under `Datasets/MI3`:

- `Datasets/MI3/derivatives/sub-011_eeg.mat` – primary training file used in the notebook.
- Raw GDF datasets for BCI Competition IV 2a/2b are stored under `Datasets/BCICIV_*` (ignored by git).

If you place the MI3 derivatives elsewhere, adjust the path in Cell 6 (“Loading the data”) of the notebook.

## Running experiments

1. **Launch VS Code / Jupyter** and open `MI3_CNN.ipynb`.
2. Run Cells 1–3 to validate the GPU environment.
3. Cell 4 defines helper functions (`data_loader`, `train_ann`, etc.). Rerun it after any edits.
4. Cell 5 instantiates the LENet and LENet_FCL architectures (PyTorch modules).
5. Cell 6 loads and balances the MI3 dataset (optional `REDUCE_REST` knob to undersample the rest class).
6. Cells 7–8 train the LENet/LENet_FCL variants. Hyperparameters (epochs, batch size, dropout) are declared at the top of each cell.
7. Cell 9 evaluates both models, and Cell 10 renders color-coded confusion matrices plus a summary table.

Weights are saved automatically via `torch.save(..., 'lenet_*.pth')` for later inference.

## Project structure

- `pyproject.toml`, `uv.lock` – dependency definitions and lockfile.
- `MI3_CNN.ipynb` – end-to-end training/evaluation notebook.
- `Datasets/` – local EEG corpora (not committed).
- `Docs/` – research notes and reference material.
- `lenet_ccb_mi3.pth`, `lenet_fcl_mi3.pth` – latest trained weights.
- `.gitignore` – excludes datasets, virtualenvs, checkpoints, and IDE artifacts.

## Next steps

- Add a scripted entry point (e.g., `train.py`) that mirrors the notebook workflow for headless runs.
- Log metrics with TensorBoard or Weights & Biases for easier experiment tracking.
- Extend evaluation to cross-subject MI3 recordings and the BCICIV 2a/2b datasets already downloaded.