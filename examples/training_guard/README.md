# Native guard development harnesses

These scripts preserve earlier source/collator/trainer validation scaffolding.
They require separately supplied pinned native assets and binding documents;
they are not a self-contained dataset/model download or the first-use example.

For a portable product integration, use the [GR00T adapter guide](../../docs/GR00T_TRAINING_GUARD.md).
For a CPU-only example requiring no external assets, use the [quick start](../../docs/QUICKSTART.md).

`portable/install.py` is retained with its original installer input controls and
regression tests. It expects a separately assembled, hash-locked native packet.
No native packet, framework environment, model weights or dataset is bundled here.
