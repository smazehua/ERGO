# ERGO licensing

This repository contains components under different licenses. The MIT grant for
original ERGO code does not relicense the third-party components listed below.

| Files or contributions | License text |
| --- | --- |
| Original ERGO code, tests, configuration, and documentation, except the files listed below | [MIT](LICENSES/MIT.txt) |
| `gs_renderer.py` and `sh_utils.py`, including their Gaussian Splatting-derived implementations | [Gaussian-Splatting License](LICENSES/Gaussian-Splatting.txt); also retain the [DreamGaussian MIT notice](LICENSES/DreamGaussian-MIT.txt) for its contributions |
| `cameras.py`, including the adapted camera conventions and ERGO modifications | [Apache-2.0 / kiuikit](LICENSES/Kiuikit-Apache-2.0.txt) |
| `zero123.py`, including the adapted component container and ERGO modifications | [Apache-2.0](LICENSES/Apache-2.0.txt) |
| `zero123plus_refine/pipeline.py`, including the adapted Zero123++ pipeline and ERGO modifications | [Apache-2.0 / Zero123++](LICENSES/Zero123Plus-Apache-2.0.txt) |

The Gaussian-Splatting License restricts use to non-commercial research and
evaluation and imposes redistribution requirements. Those restrictions continue
to apply to the derived components. The complete pipeline is therefore not
offered as an unrestricted, commercially usable MIT project. Commercial use of
restricted components requires the relevant licensors' permission.

Model weights and native third-party dependencies are not distributed in this
repository. Their licenses apply separately, including to components downloaded
using the README commands. Installing a dependency does not change its license.

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution, source links,
and a description of the adaptations. Preserve the applicable license texts and
copyright notices when redistributing these files.
