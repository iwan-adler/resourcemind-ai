# Data

Everything in `generated/` is **synthetic**, produced by `python scripts/generate_data.py`
(seed 42 by default; `--seed N` for another dataset). The app does not need these files: it generates the
identical dataset in memory. See `docs/data_dictionary.md` for columns, units and the generation model.

No real public dataset is included. To use real data, it must be appropriately licensed, converted to the
same schema and pass `resourcemind.data.validate.validate_dataset`. Then review the fairness and
under-reporting notes in `docs/assumptions_and_limitations.md` first.
