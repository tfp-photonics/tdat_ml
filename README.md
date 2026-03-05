## T-matrix database for machine learning

This repository contains scripts for machine-learning experiments on T-matrix datasets (forward prediction of T-matrices / observables from geometry and inverse prediction of geometry from T-matrices / observables). It accompanies the preprint [*A T-matrix database to promote information-driven research in nanophotonics*](https://arxiv.org/abs/2602.02101).

## Setup

Create the conda environment from `environment.yml`:

```sh
conda env create -f environment.yml
conda activate tdat_ml
```
## Data (Daphona T-matrix portal)

This code expects T-matrix datasets exported from the **Daphona T-matrix portal**:
   https://tmatrix.scc.kit.edu/

To reproduce the exact datasets used in this study,
frozen lists of dataset IDs are provided:
- `ids/cylinders.txt`
- `ids/cuboids.txt`
- `ids/cones.txt`

Use these ID lists to download the corresponding T-matrix files via the Daphona API. To avoid timeouts, download the datasets in batches (bash; Linux/macOS/WSL)::

```sh
CHUNK=24 OUTDIR="cylinders" IDSFILE="data/cylinders.txt"; mkdir -p "$OUTDIR"; TMPDIR="$(mktemp -d)"; trap 'rm -rf "$TMPDIR"' EXIT; split -l "$CHUNK" "$IDSFILE" "$TMPDIR/ids_chunk_" || { echo "split failed: check IDSFILE path or empty file"; exit 1; }; shopt -s nullglob; for f in "$TMPDIR"/ids_chunk_*; do z="$TMPDIR/$(basename "$f").zip"; echo "Downloading $(basename "$f")"; curl -sS -X POST "https://tmatrix.scc.kit.edu/api/OpenAPIExportByID/" -F "file=@$f" -F "format=zip" -o "$z" || break; unzip -oq "$z" -d "$OUTDIR" || break; rm -f "$z"; done
```

This example downloads and extracts the cylinder datasets into the folder cylinders/. Use the same command for cones and cuboids by changing OUTDIR and IDSFILE.

After downloading the data, run the Python scripts and use the Jupyter notebooks in the repository to reproduce the results and generate figures.