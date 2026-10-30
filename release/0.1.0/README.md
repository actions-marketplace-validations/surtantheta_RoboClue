# RoboClue 0.1.0

The qualified wheel is `roboclue-0.1.0-py3-none-any.whl`. Install the downloaded
wheel with Python 3.11 or newer:

```sh
python -m pip install './roboclue-0.1.0-py3-none-any.whl[dataset,mcp]'
roboclue demo
```

The source installation path is in the [project README](../../README.md).
No PyPI upload is implied.

## Qualification

The wheel SHA256 is `ab7894c4017d226e4297add0c0377b371ccac8843128929d8bb6bbe5e00e0263`.
Six complete source-suite profiles passed 6,407 tests with 63 skipped each on
Python 3.11 and 3.12. Fresh exact-wheel installed CLI/MCP, independent reviews
and supported GR00T, ACT and OpenPI native boundary checks passed.

The [qualification receipt](QUALIFICATION.json) records the exact original
source commit, wheel, criteria and terminal job results. The main qualification receipt predates
publication and is preserved unchanged. The public regression file is a
version-neutral summary that records the hash of the full original receipt,
which remains preserved locally. The [source manifest](SOURCE-MANIFEST.json)
and qualified source archive retain the exact tested source. The public README,
launch documentation and media were added after qualification; all runtime
package files are byte-identical to the qualified source. Source installation
can produce different distribution metadata because the README has changed.

[Public regression](PUBLIC-REGRESSION.json) covers 12 pinned datasets in seven
modes, for 84 cells. Sixty outputs were byte-identical. Twenty-four JSON cards
changed only the advertised version and its independently recomputed ID.
Diagnostics, statuses, exits and stderr were equal.

Twelve validated register entries ship. This evidence does not establish
physical robot outcomes, practitioner benefit, policy competence, full GR00T
fine-tune performance, Pi0 optimizer/JAX training or arbitrary distributed
backends. Native profiles and retained criteria are explicitly scoped in the
receipt and [scope guide](../../docs/SCOPE.md).

The [video review](VIDEO-REVIEW.md) applies the founder's launch brief to the
actual final media. It is an editorial review, not an external audience study.


The revised full/square launch videos use 1080p source footage from DexMimicGen.
Those adapted videos are CC-BY-SA 4.0; software licenses remain separate.
[Source and attribution](../../assets/launch/ROBOT-FOOTAGE-CREDIT.md).
The five-second opening has two slower sequential reveals, with robot playback
retimed to match. An independent [aesthetics review](AESTHETICS-REVIEW.md) was
performed alongside the VC-style editorial assessment. The project README also
includes an animated preview of the square cut; the full-resolution MP4 files
are included separately.
