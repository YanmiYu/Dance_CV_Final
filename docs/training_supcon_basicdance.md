# SupCon Basic Dance PoseGNN Training

This training surface is separate from the merged inference pipeline. It does
not change `run.py`, `configs/integrate/pipeline.yaml`, HRNet/SimpleBaseline
inference, LSTM evaluation, or report/fusion defaults.

## All-Genre c01 Config

Use:

```bash
python -m src.train.train_pose_gnn_supcon \
  --config configs/train/train_pose_gnn_supcon_basicdance_allgenre_c01.yaml
```

The all-genre config filters the PKL index to:

- `pkl_root: data/keypoints2d`
- `situation: sBM`
- `use_cameras: ["c01"]`
- `genres: all`

`cAll` files such as `gBR_sBM_cAll_d04_mBR0_ch04.pkl` are expanded into
logical camera rows, then only `c01` is retained.

## Labels

The index uses:

- `dance_label = genre + "_" + music_id + "_" + choreography_id`
- `genre_label = genre`
- `dancer = d##`
- `music_id = m...`
- `choreography_id = ch##`

SupCon positives are windows with the same `dance_label`. Different
`dance_label` windows are negatives.

## Positive Filtering

Training rows whose `dance_label` has fewer than
`min_train_rows_per_label` rows are filtered before balanced sampling. The
default is `2`. This preserves the balanced batch guarantee that every label
sampled into a batch can produce SupCon positives.

## Pair Similarity Evaluation

Validation writes retrieval metrics and explicit pair similarity metrics to:

```text
data/processed/train_pose_gnn_supcon_basicdance_allgenre_c01/metrics.json
data/processed/train_pose_gnn_supcon_basicdance_allgenre_c01/pair_similarity_eval.json
```

The pair types are:

- `same_dance_label_same_or_diff_dancer`
- `same_choreography_different_dancer`
- `same_genre_different_choreography`
- `different_genre`

Each reports mean, median, std, min, max, and pair count. Gap metrics compare
same-dance and same-choreography similarity against same-genre/different-
choreography and different-genre baselines.

## Oscar

Submit later from the repo root:

```bash
sbatch scripts/slurm/train_pose_gnn_supcon_allgenre_c01.slurm
```

The job writes checkpoints under:

```text
data/processed/train_pose_gnn_supcon_basicdance_allgenre_c01/checkpoints/
```

and deploys best checkpoint to:

```text
checkpoints/pose_gnn_encoder_basicdance_allgenre_c01_supcon.pt
```

That new checkpoint is not made the main pipeline default. The old checkpoint
`checkpoints/pose_gnn_encoder_oscar.pt` remains the integrated default.
