# Supervised Contrastive Training — Basic Dance Embedding

This pipeline trains a stronger choreography-aware pose embedding on AIST++
Basic Dance (`situation == sBM`) using supervised contrastive (SupCon) loss.
Windows from the same `dance_label = genre + "_" + music_id + "_" +
choreography_id` are pulled together; windows from different labels are
pushed apart.

## Components

| File | Purpose |
| --- | --- |
| `src/datasets/basic_dance_index.py` | Parse PKL filenames, expand `cAll` files into per-camera rows, write `data/processed/basic_dance_embedding_index.csv`. |
| `src/datasets/basic_dance_supcon_dataset.py` | Temporal-window dataset built on top of the index. |
| `src/datasets/balanced_batch_sampler.py` | `DanceLabelBalancedBatchSampler` — N labels × K samples per batch. |
| `src/models/pose_gnn_temporal.py` | `PoseGNNTemporalEncoder` — wraps `PoseGNNEncoder`, mean-pools over time, projects, L2-normalises. |
| `src/losses/supcon.py` | `SupConLoss` (cosine formulation, safe for empty-positive anchors). |
| `src/train/train_pose_gnn_supcon.py` | CLI training script. |
| `configs/train/train_pose_gnn_supcon_basicdance.yaml` | Default config. |
| `scripts/slurm/train_pose_gnn_supcon_basicdance.slurm` | Oscar SLURM job. |

## Data

The expected primary layout is `data/keypoints2d/` with consolidated
`cAll` files such as

    gBR_sBM_cAll_d04_mBR0_ch01.pkl

containing `keypoints2d` of shape `(num_cameras, T, 17, 3)` (x, y, conf).
The index builder expands each consolidated PKL into one logical row per
camera (`c01`..`c09`). The legacy per-camera layout under
`data/labels/aistpp/keypoints2d_raw/` is also supported (each file is a
single `(T, 17, 3)` array).

## 1. Build the index

```bash
python -m src.datasets.basic_dance_index \
  --pkl-root data/keypoints2d \
  --out-csv  data/processed/basic_dance_embedding_index.csv \
  --situation sBM
```

The training script will build the index automatically the first time it
runs if the CSV is missing.

## 2. Train locally

```bash
python -m src.train.train_pose_gnn_supcon \
  --config configs/train/train_pose_gnn_supcon_basicdance.yaml
```

Outputs land in `data/processed/train_pose_gnn_supcon_basicdance/`:

```
checkpoints/
  best.pt            # lowest val SupCon loss
  last.pt            # last epoch
  epoch_XXX.pt       # periodic checkpoints (configurable)
metrics.json         # best-epoch summary + resolved config
train_log.csv        # per-epoch row of metrics
config_resolved.json # the merged runtime config
```

The best checkpoint is also copied to
`checkpoints/pose_gnn_encoder_basicdance_supcon.pt` (set
`deploy_checkpoint: null` in the YAML to disable).

## 3. Train on Oscar

```bash
sbatch scripts/slurm/train_pose_gnn_supcon_basicdance.slurm
```

Override the config or pass extra args:

```bash
sbatch --export=ALL,TRAIN_CFG=configs/train/train_pose_gnn_supcon_basicdance.yaml,EXTRA_ARGS="--epochs 30" \
       scripts/slurm/train_pose_gnn_supcon_basicdance.slurm
```

## 4. Evaluation metrics

Reported every `eval_every` epochs, computed on the held-out validation set:

* `val_supcon_loss`
* `val_mean_pos_cos` / `val_mean_neg_cos` (cosine sim of in-batch + / - pairs)
* `val_retrieval_top1_dance_label` / `val_retrieval_top5_dance_label`
* `val_retrieval_top1_genre`       / `val_retrieval_top5_genre`

Retrieval is computed by encoding all val windows, scoring the cosine
similarity matrix, and asking whether the top-k nearest neighbours
(excluding self) contain the same label.

## 5. Use the new checkpoint in `render_report`

The checkpoint is saved with the frame-encoder weights under the
legacy `"model"` key, so the existing report loader works unchanged:

```bash
python -m src.compare.render_report \
  --benchmark <bench.mp4> --user <user.mp4> \
  --model-config configs/model/hrnet_w32.yaml \
  --ckpt        checkpoints/pose_aist_only.pt \
  --alignment-method gnn_embedding \
  --gnn-checkpoint checkpoints/pose_gnn_encoder_basicdance_supcon.pt
```

`src/compare/embedding_features.py:load_pose_gnn_encoder` accepts:

* a raw `state_dict` (legacy triplet checkpoints), or
* a dict with `model` + `embedding_dim` (legacy + new format), or
* a SupCon dict that *also* has `temporal_model` (the temporal head is
  ignored; only the frame encoder is used by `render_report`).

## Split modes

* `split_mode: dance_label` (default) — distinct dance_label values are
  partitioned between train and val. Tests generalisation to **unseen
  choreographies**.
* `split_mode: camera` — same dance_label may appear in both splits but
  with disjoint cameras. Tests **view invariance**. Set `val_cameras`
  explicitly (e.g. `["c08", "c09"]`).

## Notes on the filtered data

`data/keypoints2d/` currently contains 121 consolidated `cAll` PKLs
covering only `gBR_sBM` (Break Basic). With the per-camera expansion that
gives ~1089 logical samples and ~60 distinct `dance_label` classes — enough
to train SupCon, though the model will only see one genre. To learn a
*genre-aware* embedding (so retrieval_top1_genre is meaningful) you'll
want PKLs from the other AIST++ genres (`gHO`, `gKR`, `gJB`, ...) added
to the same folder. The index builder and dataset already handle the
multi-genre case; no code changes needed.
