# Benchmark / Imitation Recording Protocol

All paired clips that we record ourselves (benchmark + imitation) must follow
this protocol so that the scoring pipeline is comparing like for like.

## Framing

- Camera at chest / waist height.
- Full body visible end to end (head to feet), including during extreme moves.
- Stable tripod. No handheld.
- No zoom during the clip.
- Consistent distance from camera across benchmark and imitation takes.
- Plain / uncluttered background when possible.

## Clip bounds

- 10-20 seconds per clip.
- Benchmark and imitation clips are trimmed to the SAME choreographic phrase.
- If in doubt, cut at a clear beat transition.

## Capture settings

- Same frame rate between benchmark and imitation when possible (30 fps preferred).
- Same resolution and aspect ratio when possible.
- Good lighting; avoid harsh backlighting.

## Naming convention

`<song_id>_<phrase_id>_<role>_<take>_<cam>.mp4`

Examples:
- `blackpink_01_benchmark_take1_camA.mp4`
- `blackpink_01_userA_take2_camA.mp4`

Where:
- `song_id`: short slug, e.g. `blackpink`, `doja_paint_the_town`.
- `phrase_id`: two-digit id for the choreography phrase (`01`, `02`, ...).
- `role`: `benchmark` or `userX` (`userA`, `userB`, ...).
- `take`: `take1`, `take2`, ...
- `cam`: `camA`, `camB`, ... (matches `camera_id` in the manifest).

## Pair manifest fields

Stored in `data/manifests/pair_manifest.parquet` (or `.csv`). Required columns:

| column        | meaning                                         |
|---------------|-------------------------------------------------|
| pair_id       | unique id linking benchmark + imitation(s)      |
| song_id       | matches naming convention                       |
| phrase_id     | matches naming convention                       |
| role          | `benchmark` or `userX`                          |
| performer_id  | e.g. `userA`, or `benchmark` for the reference   |
| camera_id     | `camA`, `camB`, ...                             |
| path          | local path to the trimmed clip                  |
| start_sec     | optional trim start inside the path file        |
| end_sec       | optional trim end inside the path file          |

See `src/data/build_manifest.py::build_pair_manifest`.
