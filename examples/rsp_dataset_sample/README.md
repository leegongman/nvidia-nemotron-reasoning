# RSP Dataset Sample

This directory contains tiny schema examples for the three RSP row families.
They are not a trainable dataset and will not pass the full row-count gates in
`scripts/data/verify_rsp_dataset.py`.

For real training, stage the full/private dataset under:

```text
data/rsp_dataset/
```

Expected full files:

- `rsp_anchor_sft.jsonl`
- `rsp_decision_sft.jsonl`
- `rsp_decision_preferences.jsonl`
- `rsp_manifest.json`
