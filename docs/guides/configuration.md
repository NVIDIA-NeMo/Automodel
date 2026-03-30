# YAML Configuration

NeMo AutoModel recipes are configured with YAML. Under the hood, YAML is parsed into a `ConfigNode` which:

- Translates common scalar strings into typed Python values (e.g., `"10"` → `10`).
- Resolves `_target_` (and `*_fn`) into Python callables/classes.
- Supports environment variable interpolation inside YAML strings.
- Tries to make config printing safe by preserving original placeholders (to avoid leaking secrets).


## Load Model and Dataset Configs

Most recipes load the YAML using `nemo_automodel.components.config.loader.load_yaml_config()`, which returns a `ConfigNode`.

Within a `ConfigNode`:

- Nested dicts become nested `ConfigNode` objects.
- Lists are recursively wrapped.
- Scalars are translated with `translate_value()` when they are YAML strings.

### Typed Scalar Translation (`translate_value`)

Only **strings** are translated. Examples:

- `"123"` → `123`
- `"3.14"` → `3.14`
- `"true"` / `"false"` → `True` / `False`
- `"None"` / `"none"` → `None`

YAML-native types (like `step_size: 10` without quotes) are already typed by the YAML parser and remain unchanged.


## Use `_target_` for Instantiation

Any mapping containing a `_target_` key can be instantiated using `ConfigNode.instantiate()`:

```yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: meta-llama/Llama-3.2-1B
```

There is also support for resolving callables from:

- **Dotted paths**: `pkg.module.symbol`
- **Local file paths**: `/abs/path/to/file.py:symbol`

### Safety and Policy

By default, resolving targets is restricted:

- Imports are allowed from common safe prefixes (e.g. `nemo_automodel`, `torch`, `transformers`, …).
- Accessing private or dunder attributes is blocked by default.
- Loading out-of-tree user code can be enabled with `NEMO_ENABLE_USER_MODULES=1` or by calling `set_enable_user_modules(True)`.

## Distributed Section (Strategy-Based)

The `distributed:` section is **not** instantiated using `_target_`. Recipes parse it with a fixed schema: use `strategy: fsdp2`, `strategy: ddp`, or `strategy: megatron_fsdp`, plus optional parallelism sizes (`dp_size`, `tp_size`, `pp_size`, etc.) and strategy-specific options. When pipeline parallelism is enabled (`pp_size > 1`), add a `pipeline:` subsection with options such as `pp_schedule`, `pp_microbatch_size`, and `layers_per_stage`. See the [Pipelining](pipelining.md) guide and recipe example configs for full examples.


## Interpolate Environment Variables in YAML

NeMo AutoModel supports env var interpolation inside YAML **string values**.

### Supported Forms

- **Braced**:
  - `${VAR}`
  - `${VAR,default}`
  - `${var.dot.var}` (dots are treated as part of the env var name)
- **Dollar**:
  - `$VAR`
  - `$var.dot.var`
- **Back-compat**:
  - `${oc.env:VAR}`
  - `${oc.env:VAR,default}`

### Interpolation Behavior

- Interpolation happens when values are wrapped into a `ConfigNode`.
- If a referenced env var is **missing** and **no default** is provided, config loading raises a `KeyError`.
- Defaults are supported only for braced forms using the first comma: `${VAR,default_value}`.

### Example (Databricks Delta)

```yaml
dataset:
  _target_: nemo_automodel.components.datasets.llm.column_mapped_text_instruction_iterable_dataset.ColumnMappedTextInstructionIterableDataset
  path_or_dataset_id: delta://catalog.schema.training_data
  delta_storage_options:
    DATABRICKS_HOST: ${DATABRICKS_HOST}
    DATABRICKS_TOKEN: ${DATABRICKS_TOKEN}
    DATABRICKS_HTTP_PATH: ${DATABRICKS_HTTP_PATH}
```


## Prevent Secret Leakage in Logs

When an env var placeholder is resolved, the config keeps the original placeholder in an internal `._orig_value` field for **safe printing**:

- `str(cfg)` / `repr(cfg)` prints placeholders (e.g. `${DATABRICKS_TOKEN}`), not resolved secrets.
- `cfg.to_yaml_dict(use_orig_values=True, redact_sensitive=True)` is the recommended way to produce a loggable YAML dict.

:::{important}
Printing a **leaf value** (for example, `print(cfg.dataset.delta_storage_options.DATABRICKS_TOKEN)`) outputs the resolved secret. Instead, print the full config or use a redacted YAML dict.
:::


## Configure Slurm

SLURM jobs are submitted with `sbatch` directly — no YAML section needed.
Copy the reference script, edit `CONFIG` and cluster settings, then submit:

```bash
cp slurm.sub my_cluster.sub
vim my_cluster.sub
sbatch my_cluster.sub
```

All cluster-specific configuration (SBATCH directives, container image, mounts,
secrets, environment variables) lives in your sbatch script.  See
[Run on a Cluster](../launcher/slurm.md) for full examples.

