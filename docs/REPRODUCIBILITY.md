# Reproducibility

- Set seeds via `odd.utils.set_seed`.
- Enable PyTorch deterministic algorithms with `--deterministic` for exact repeatability (slower).
- Record environment with `pip freeze > requirements-lock.txt`.
- Use `configs/default.yaml` for baseline hyperparameters.
