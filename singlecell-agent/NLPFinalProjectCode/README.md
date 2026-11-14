# NLP Final Project Code

This folder contains the final notebook, the local dataset, and the Python package list in `requirements.txt`.

## Recreate the Python Environment

If you want to build a new environment from the package list, use `pip` with `requirements.txt`.
The full `requirements.txt` came from a broader class environment, so it may include packages beyond this single notebook.

### Option 1: Conda

```bash
conda create -n nlpfinal python=3.13 -y
conda activate nlpfinal
pip install -r requirements.txt
```

### Option 2: venv

```bash
python -m venv nlpfinal
source nlpfinal/bin/activate
pip install -r requirements.txt
```

## Open the Notebook

From this folder, start Jupyter:

```bash
jupyter notebook
```

Then open `FinalProjectNotebook.ipynb`.

## Use in Google Colab

For Colab, use `requirements-colab.txt` instead of the full `requirements.txt`.

1. Upload the entire `NLPFinalProjectCode` folder to Google Drive.
2. Open `FinalProjectNotebook.ipynb` in Colab.
3. In the first setup cell, set `PROJECT_DIR` to your Drive folder path if needed.
4. Run the setup cell to mount Drive and install the Colab package list:

```bash
pip install -r requirements-colab.txt
```

The notebook now includes a Colab setup section and expects the data to stay inside `data/` within `NLPFinalProjectCode`.

## Notes

- Run the install command from this folder so `requirements.txt` is found correctly.
- The notebook expects the project data to be inside `NLPFinalProjectCode/data/`.
- If you use Conda, `pip install -r requirements.txt` should be run after activating the new Conda environment.
