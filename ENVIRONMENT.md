# Environment

Environment: `finder-paper`. Requires Linux, Bash, and an NVIDIA driver compatible with CUDA 12.8.

```bash
conda env create -f environment.yml
conda activate finder-paper
# Update:
conda env update -n finder-paper -f environment.yml --prune
```

| Package | Version | Installer / source | Use |
|---|---|---|---|
| Python | 3.10.21 | Conda / conda-forge | Runtime |
| pip | Conda-resolved | Conda / conda-forge | Package installation |
| torch | 2.8.0+cu128 | pip / download.pytorch.org, cu128 | GPU training |
| numpy | 2.1.2 | pip / PyPI | Arrays |
| scipy | 1.15.3 | pip / PyPI | Sparse matrices and statistics |
| pandas | 2.3.2 | pip / PyPI | Tables |
| scikit-learn | 1.7.1 | pip / PyPI | Splits and metrics |
| torch-geometric | 2.6.1 | pip / PyPI | Graph layers |
| torch_scatter | 2.1.2+pt28cu128 | pip / data.pyg.org | Graph operations |
| torch_sparse | 0.6.18+pt28cu128 | pip / data.pyg.org | Sparse graph operations |
| matplotlib | 3.10.9 | pip / PyPI | Plots |
| seaborn | 0.13.2 | pip / PyPI | Plots |
| tqdm | 4.70.0 | pip / PyPI | Progress bars |
| networkx | 3.3 | pip / PyPI | Graph utilities |

Additional preprocessing dependencies, not included in `environment.yml`:

| Package | Version | Installer / source | Use |
|---|---|---|---|
| rdkit | Not pinned | Conda / conda-forge | Molecular graphs |
| transformers | Not pinned | pip / PyPI | BioBERT |
| pyarrow | Not pinned | Conda / conda-forge | Parquet files |

Record chosen preprocessing versions in a separate environment manifest. Bash is a system dependency; NVIDIA drivers are supplied by NVIDIA.

## Cleanup

Remove the environment when deleting the project. Check other projects before updating or removing a shared environment.

```bash
conda deactivate
conda env remove -n finder-paper
```
