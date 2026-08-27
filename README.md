# ChemNetwork

A small **flow matching** playground: a neural network learns to transport a 2D Gaussian
point cloud into a target distribution, served through a FastAPI app with a browser
frontend that visualises the whole thing — source cloud, generated cloud, and the animated
ODE trajectory in between.

> [!NOTE]
> **This is a sandbox project.** It exists to learn FastAPI, PyTorch, Hydra, Weights &
> Biases and Docker on a problem small enough to stay debuggable — 2D toy distributions
> that train in minutes on a laptop CPU. The name is aspirational: the real goal is a
> network trained on actual (bio)chemical data from PubChem/ChEMBL to predict molecular
> properties. That model is **in the pipeline**; everything here is the scaffolding being
> built and tested first. Nothing in this repo does chemistry yet.

---

## What it actually does

Flow matching learns a time-dependent velocity field $v_\theta(x, t)$ that transports a
simple source distribution $p_0$ into a complicated target $p_1$.

**Training** ([`chemnetwork/training/train.py`](chemnetwork/training/train.py)) — for each step:

1. Draw a batch $x_0 \sim p_0$ (2D standard Gaussian) and $x_1 \sim p_1$ (the target).
2. Sample $t \sim \mathcal{U}(0,1)$ and linearly interpolate: $x_t = t\,x_1 + (1-t)\,x_0$.
3. The regression target is the conditional flow $u_t = x_1 - x_0$.
4. An MLP predicts $v_\theta(x_t, t)$; minimise $\lVert v_\theta - u_t \rVert^2$.

**Sampling** ([`chemnetwork/sample.py`](chemnetwork/sample.py)) — draw $x_0 \sim p_0$ and
integrate $\dot{x} = v_\theta(x, t)$ from $t=0$ to $t=1$ with an Euler solver. Optionally
keep every intermediate step, which is what the frontend animates.

The MLP is deliberately tiny: 3 inputs `(t, x, y)` → 3×128 hidden ReLU layers → 2 outputs.

### Target distributions

Two are implemented, each with its own trained checkpoint:

| Target | Description |
| --- | --- |
| `moons` | Two interleaving half-circles (`sklearn.datasets.make_moons`), with noise |
| `checkerboard` | 4×4 checkerboard of uniform squares |

Adding a third is a contained change — see [Adding a target distribution](#adding-a-target-distribution).


## Quickstart

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                    # create .venv and install dependencies
```

**Train a model first.** Checkpoints are gitignored, so a fresh clone has none and the
server will fail at startup with a `FileNotFoundError` if it can't find the files listed in
`checkpoint_paths`. Training is fast — roughly a minute per target on a laptop CPU
(~940 it/s for the default 50k steps):

```bash
# Train both implemented targets, ~2 minutes total
uv run python -m chemnetwork.training.train data.target_distribution=moons \
    wandb.mode=disabled train_parameters.comment=moons
uv run python -m chemnetwork.training.train data.target_distribution=checkerboard \
    wandb.mode=disabled train_parameters.comment=checkerboard

# Put the final checkpoints where the app expects them
mkdir -p serving_checkpoint
cp runs/*-moons/checkpoints/*_step_50000.pt serving_checkpoint/
cp runs/*-checkerboard/checkpoints/*_step_50000.pt serving_checkpoint/
```

Then serve:

```bash
uv run uvicorn app.main:app --reload       # http://localhost:8000
```

Open <http://localhost:8000> — the frontend is served from the app root.

With Docker instead (still needs the checkpoints in place first):

```bash
docker compose up --build                  # also http://localhost:8000
```

### Using the web UI

Pick a **target distribution**, set the number of **samples** (1–5000) and **integration
steps** (2–1000), then hit **Generate**. You get three plots:

- **Source distribution** — the Gaussian the samples started as
- **Generated distribution** — the model's output, overlaid on ground-truth target samples so you can eyeball the fit
- **Trajectory animation** — the point cloud morphing from source to target, replayable with the ▶ button

Fewer integration steps means a coarser Euler approximation — a good knob to watch the
generated cloud degrade. The animation is downsampled to at most
`max_trajectory_steps` frames regardless of how many solver steps were actually taken.

## HTTP API

Interactive docs at <http://localhost:8000/docs>.

### `GET /health`

```json
{ "status": "ok" }
```

### `GET /available`

Lists the targets that have a served model, plus the one the frontend preselects.

```bash
curl http://localhost:8000/available
```

```json
{
  "targets": [
    { "id": "checkerboard", "label": "Checkerboard" },
    { "id": "moons",        "label": "Two moons" }
  ],
  "default": "moons"
}
```

`id` is the stable machine key sent back in `/generate`; `label` is display text only.

### `POST /generate`

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"num_samples": 500, "integration_steps": 50, "target": "moons", "return_trajectory": true}'
```

**Request**

| Field | Type | Default | Constraints |
| --- | --- | --- | --- |
| `num_samples` | int | `500` | 1 – 5000 |
| `integration_steps` | int | `100` | 2 – 1000 |
| `target` | enum | `moons` | `moons` \| `checkerboard` |
| `return_trajectory` | bool | `false` | — |

**Response**

| Field | Shape | Meaning |
| --- | --- | --- |
| `num_samples` | int | Echo of the request |
| `target` | enum | Echo of the request |
| `source_points` | `(N, 2)` | Samples drawn from $p_0$ |
| `generated_points` | `(T, N, 2)` | Trajectory. `T = 1` when `return_trajectory` is false |
| `target_points` | `(N, 2)` | Ground-truth samples from $p_1$, for visual comparison |

Coordinates are rounded to 3 decimals to keep payloads small.

**Errors** are split by cause:

| Status | Cause |
| --- | --- |
| `422` | Malformed request — unknown target name, or a value outside its bounds |
| `404` | Valid target name, but no checkpoint for it is being served |

---

## Training your own model

Training is driven by [Hydra](https://hydra.cc/), with configs in
[`preferences/`](preferences/). Any value can be overridden on the command line:

```bash
# Default run (moons)
uv run python -m chemnetwork.training.train train_parameters.comment=my_run

# Train the checkerboard target for longer, with a bigger network
uv run python -m chemnetwork.training.train \
    data.target_distribution=checkerboard \
    train_parameters.train_loop_iterations=100001 \
    model.params.hidden_features=256 \
    train_parameters.comment=checkerboard_big
```

Each run gets its own directory `runs/<timestamp>-<comment>/`, containing Hydra's resolved
config, logs, and a `checkpoints/` folder. Checkpoints are written every
`train_parameters.save_every` steps and named
`<model>_<target>_weights_step_<step>.pt`.

### Configuration groups

| File | Controls |
| --- | --- |
| [`preferences/config.yaml`](preferences/config.yaml) | Composition root, W&B settings, run directory layout |
| [`preferences/data/flow_matching_data_param.yaml`](preferences/data/flow_matching_data_param.yaml) | `target_distribution`, `target_data_noise`, `train_batch_size` |
| [`preferences/model/mlp.yaml`](preferences/model/mlp.yaml) | Layer widths |
| [`preferences/train_parameters/flow_matching_parameters.yaml`](preferences/train_parameters/flow_matching_parameters.yaml) | Iterations, learning rate, seed, checkpoint interval |

Training logs to Weights & Biases by default. To run without it:

```bash
uv run python -m chemnetwork.training.train wandb.mode=disabled
```

### Serving a newly trained model

1. Copy the checkpoint into `serving_checkpoint/`.
2. Add its path to `checkpoint_paths` in [`app/config.py`](app/config.py), or override at
   runtime (see below).
3. Restart the server. The new target appears in the dropdown automatically — the
   registry reads the target name out of the checkpoint's embedded config.

Two checkpoints claiming the *same* target, or one claiming an unknown target name, fail
loudly at startup rather than at request time.

---

## Runtime configuration

Settings live in [`app/config.py`](app/config.py) and can be overridden by environment
variables with a `CHEMNET_` prefix.

| Setting | Env var | Default |
| --- | --- | --- |
| `checkpoint_paths` | `CHEMNET_CHECKPOINT_PATHS` | the two `serving_checkpoint/` paths |
| `default_target` | `CHEMNET_DEFAULT_TARGET` | `moons` |
| `max_trajectory_steps` | `CHEMNET_MAX_TRAJECTORY_STEPS` | `50` |

List values are parsed as JSON:

```bash
CHEMNET_CHECKPOINT_PATHS='["serving_checkpoint/MultiLayerPerceptron_moons_weights_step_50000.pt"]' \
CHEMNET_DEFAULT_TARGET=moons \
uv run uvicorn app.main:app
```

Startup fails fast if `default_target` has no served model — the frontend preselects that
value, so a misconfigured default must not turn into a 404 on the user's first click.

---

## Adding a target distribution

The seam is deliberately narrow. To add, say, a spiral:

1. **Add the enum member** in [`chemnetwork/data/point_clouds.py`](chemnetwork/data/point_clouds.py):
   ```python
   class TargetDistribution(str, Enum):
       MOONS = "moons"
       CHECKERBOARD = "checkerboard"
       SPIRAL = "spiral"
   ```
2. **Implement `_draw_spiral`** on `PointCloudGenerator`, returning a `(num_samples, 2)`
   float32 tensor, and register it in the `draw_target` dispatch dict.
3. **Add a display label** to `TARGET_LABELS` in [`app/schemas.py`](app/schemas.py).
   Optional — an unlabelled target falls back to a prettified version of its value.
4. **Train it**: `... train data.target_distribution=spiral`
5. **Serve it**: copy the checkpoint and add its path to `checkpoint_paths`.

No frontend changes are needed. The dropdown is populated from `/available` at page load.

---

## Project layout

```
app/                         FastAPI serving layer
├── main.py                  App entry point; loads the registry on startup
├── config.py                Pydantic settings (env-overridable)
├── schemas.py               Request/response models and display labels
├── model_registry.py        Maps target name -> (model, training config)
├── dependencies.py          Registry injection for routes
├── serialization.py         Tensor -> JSON helpers (downsampling, rounding)
├── routers/samples.py       /health, /available, /generate
└── static/                  Frontend (vanilla JS + Plotly, no build step)

chemnetwork/                 The ML core, independent of the web layer
├── models/flow_matching.py  FlowModel, FlowMatcher, MSELoss, NumericalODESolver
├── data/point_clouds.py     Source and target distribution sampling
├── training/train.py        Hydra-driven training loop
├── sample.py                Checkpoint loading and sample generation
└── utils.py                 Seeding

preferences/                 Hydra config groups
serving_checkpoint/          Checkpoints the app serves (gitignored)
runs/                        Training outputs, one dir per run (gitignored)
tests/unit/                  Pytest suite
```

The split matters: `chemnetwork/` knows nothing about HTTP, and `app/` treats the model as
a black box behind the registry. Either can be exercised without the other.

---

## Tests

```bash
uv run pytest              # whole suite
uv run pytest tests/unit/app -v
```

Covers the flow matching components, the ODE solver, distribution sampling, the
serialization helpers and the model registry.

---

## Frontend notes

Deliberately dependency-free — vanilla ES modules plus Plotly from a CDN, no build step,
no framework. Worth knowing if you touch it:

- The parameter panel is a real `<form>`, so the browser handles constraint validation and
  Enter-to-submit; the JS handler only runs on an already-valid form.
- The target dropdown is populated at load from `/available`, in an `init()` that runs
  *after* the event listeners are registered — a failed request shows an error and disables
  the button instead of leaving a silently dead page.
- Styling uses CSS custom properties with a dark-mode block that redefines only the colour
  tokens, so no component rule needs a dark variant.

---

## Roadmap

- [ ] **The actual point:** a network trained on real (bio)chemical data (PubChem/ChEMBL)
      for molecular property prediction
- [ ] Higher-order ODE solvers — `_midpoint_solver` and `_rk4_solver` are stubs that raise
      `NotImplementedError`
- [ ] Per-target plot ranges instead of a hard-coded `[-4, 4]`
- [ ] Expose `return_trajectory` in the UI
- [ ] Versioned API prefix (`/api/v1`)
