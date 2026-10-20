# Security and execution scope

ERGO is a local research pipeline for Linux/CUDA. It does not provide an upload
service or an HTTP inference server. Run it with trusted inputs, configuration,
dependencies, and model snapshots, using a dedicated environment.

## Model loading

Zero123-XL loading enables `trust_remote_code=True` for custom components. Local
or offline loading is not a sandbox: model-supplied Python code can execute.
Review custom code and verify snapshot revisions/checksums before use. Keep
model directories read-only during runs. Safetensors weights, when available,
avoid pickle deserialization but do not make custom Python components safe.

## Dependency baseline

`requirements.txt` records the previously tested research environment. It is not
a guarantee that these versions are free of known vulnerabilities. Relevant
published advisories include:

| Dependency | Advisory and conditions |
| --- | --- |
| PyTorch 2.7.1 | [CVE-2026-24747](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p): malicious pickle checkpoints can corrupt memory even with `weights_only=True`. Do not load untrusted checkpoints. |
| Accelerate 1.14.0 | [Checkpoint path traversal / blocking report](https://github.com/huggingface/accelerate/issues/4067): malicious shard indexes can escape the model directory or reference blocking files. The upstream maintainers dispute its security classification. |
| rembg 2.0.69 | [Custom-model path traversal](https://github.com/danielgatis/rembg/security/advisories/GHSA-3wqj-33cg-xc48) and [SSRF/CORS](https://github.com/danielgatis/rembg/security/advisories/GHSA-55v6-g8pm-pw4c) affect server usage. ERGO calls the fixed U2-Net library session and does not start that server. |

These are dependency advisories, not proof that every issue is reachable through
ERGO's current workflow. Revalidate model loading and rebuild CUDA extensions
when changing PyTorch/CUDA or native dependencies; a fresh GPU installation has
not been independently verified for this source release.

## Run state and reporting

`run_manifest.json` and stage hash receipts prevent reuse of mismatched outputs;
they are operational state. Model content is not hashed by the runner. Use a new
output directory whenever model contents change, even if their path is unchanged.
The manifest can contain an absolute local model path; review it before sharing.

For a vulnerability report, use GitHub private vulnerability reporting if enabled
for this repository. Avoid posting sensitive model code, credentials, or local
file contents in a public issue.
