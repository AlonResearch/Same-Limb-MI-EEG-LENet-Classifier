# MI-EEG Motor Imagery Classification

Motor-imagery EEG trials from the MI3 dataset are classified with lightweight LENet variants. Two models are available:

- **LENet (Conv-only classifier block)** – matches the MI2/BCI IV-2a architecture in the original publication.
- **LENet_FCL** – keeps the convolutional front-end but swaps the final block for a dynamically sized fully connected layer to support higher-channel MI3 recordings.
- More models (RNNs, Transformers, etc) to be added soon...

Training, evaluation, and visualization live in `Notebooks/'datasetname'_'modelname'.ipynb`. Saved weights for the most recent training run of each model are stored under `ModelWeights/` as `'ModelName'_'DatasetName'.pth`.

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

## Preprocessing scripts

- Raw-to-mat conversion lives in the MATLAB `.m` utilities bundled with each dataset folder (see [Datasets/BCICIV_2a_gdf/Code](Datasets/BCICIV_2a_gdf/Code) and [Datasets/MI3/code](Datasets/MI3/code)).

- Dataset-specific READMEs inside `Datasets/**` document channel mapping, filtering, and export parameters (e.g., [Datasets/MI3/README.md](Datasets/MI3/README.txt)). Follow those instructions before running the notebook so the derivatives match the expected format.

## Running experiments

1. **Launch VS Code / Jupyter** and open `Notebooks/MI3_CNN.ipynb`.
2. Run Cells 1–3 to validate the GPU environment.
3. Cell 4 defines helper functions (`data_loader`, `train_ann`, etc.). Rerun it after any edits.
4. Cell 5 instantiates the LENet and LENet_FCL architectures (PyTorch modules).
5. Cell 6 loads and balances the MI3 dataset (optional `REDUCE_REST` knob to undersample the rest class).
6. Cells 7–8 train the LENet/LENet_FCL variants. Hyperparameters (epochs, batch size, dropout) are declared at the top of each cell.
7. Cell 9 evaluates both models, and Cell 10 renders color-coded confusion matrices plus a summary table.

Weights are saved automatically via `torch.save(..., 'ModelWeights/lenet_*.pth')` for later inference.

## Next steps

- Add new ML models and try to surpass the 90% accuracy
- Use transfer learning
- Use atention mechanisms
- Use a variable learning ratio depending on the size of the loss
- For CNNs change the number and size of kernels, temporal and spatial convolution blocks
