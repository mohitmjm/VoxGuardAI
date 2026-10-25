# Research validation acquisition: checked 2026-10-25

## Access and rights findings

The workspace had approximately **28 GiB available** before investigation. No full corpus was downloaded. All checks used configured proxy routes; no network-policy bypass, alternate IP, credential extraction or TLS relaxation was attempted.

| Candidate/source | Actual check | Outcome and research consequence |
|---|---|---|
| ASVspoof 5 official challenge | `https://www.asvspoof.org/index2024.html` | Proxy tunnel HTTP 403. Release recording rights/download protocol not available; no recordings acquired. |
| ASVspoof 5 authors' repository | `https://github.com/asvspoof-challenge/asvspoof5` | Git clone succeeds at `fe23d3053a0889b305d98c9e4cadf418c213d972`. Official baseline/evaluation source is inspectable, but repository contains no audio corpus. Official Releases page is accessible and reports no releases/download assets. Its software license is not dataset permission. |
| In-the-Wild authoritative host | `https://deepfake-demo.aisec.fraunhofer.de/in_the_wild` | Proxy tunnel HTTP 403. Recording rights and current metadata were not retrieved. |
| WaveFake official distribution | `https://zenodo.org/record/5642694`, linked by authors | Zenodo API/site proxy denied. Dataset-specific license not retrieved. |
| WaveFake authors' repository | `https://github.com/RUB-SysSec/WaveFake` | Git clone succeeds at `d52d51b9ccdb0cec3f484e84b228791f06b955be`; official Releases page reports no releases/download assets. Five LJSpeech and five generated WAV test fixtures exist and test utilities label their directories real/fake. Only an MIT **software** license was found, without explicit fixture recording rights or synthesis-family metadata. These fixtures were **not used for new evaluation**; permissions were not inferred. |
| MLAAD authoritative project | `https://deepfake-total.com/mlaad` | Proxy tunnel HTTP 403; no recording rights or corpus acquired. Unknown third-party GitHub rehosts are not treated as authorized mirrors. |
| Hugging Face potential official distribution | `https://huggingface.co/datasets/asvspoof-challenge/asvspoof5` and public dataset search API | Proxy tunnel HTTP 403; no gated/third-party mirror was downloaded. |
| GitHub REST metadata | `https://api.github.com/...` | Proxy tunnel HTTP 403. Normal Git/raw GitHub requests work; they were used only to inspect authors' public sources, not as proxies to blocked content. |

This check found **no independently licensed modern-synthesis subset suitable for representative validation or untouched testing**. The existing six-recording LibriSpeech/Flite pilot remains a separate engineering exercise. No new benchmark metrics or modern-generator capability can be claimed from source-code inspection.

Authoritative recording hosts requiring access for further work: `www.asvspoof.org`, `zenodo.org`, `deepfake-demo.aisec.fraunhofer.de`, `deepfake-total.com`, and an official verified Hugging Face distribution if applicable. Access alone is not recording permission. A user may instead acquire a permitted release locally and import its original protocol below. Never ask for raw credentials in chat.

## Bounded resumable public downloads

`scripts/datasets/download.py` downloads one explicitly chosen public HTTPS object, not a whole dataset. It requires retained reviewed recording rights, an authoritative rights URL, expected SHA-256, and a maximum byte count. It retains `.part` bytes on interruption; resumes only matching source/rights/hash provenance; validates Content-Range/ETag when provided; enforces storage/size bounds; and publishes the final file only after streaming checksum verification. It restarts safely if the server ignores Range. No archives are extracted. Redirects to new hosts require an explicit reviewed hostname. Private/signed/query-bearing URLs are intentionally unsupported to keep credentials out of provenance files.

Template (replace every release-specific argument with reviewed authoritative values):

```powershell
.\.venv\Scripts\python.exe -m scripts.datasets.download --url 'https://OFFICIAL-HOST/RELEASE-FILE' --output artifacts/datasets/release.part-download --sha256 OFFICIAL_SHA256 --max-bytes EXPECTED_MAX_BYTES --license-file data/REVIEWED_RECORDING_LICENSE.txt --license-url 'https://OFFICIAL-HOST/RECORDING-TERMS' --rights-reviewed
```

If no author checksum exists, record a trusted separately acquired release digest and its provenance before using this helper; do not invent or change a checksum to make a download pass. This tool cannot defeat access denial or create a license.

## Manual local import and partial corpus evaluation

Acquire the release through a permitted official route on a machine with access, retain terms, and extract using a trusted archive tool. Keep raw recordings outside Git. Prefer a normalized CSV with columns:

`audio_path,label,speaker_id,source_id,generator,split,language`

- `audio_path` is relative to `--audio-root`; traversal and symlinks escaping this root are rejected.
- Use official speaker and original utterance/source identifiers. Missing/unknown speaker/source grouping is rejected rather than guessed from filenames. Obtain a reviewed provenance mapping when the protocol omits source groups.
- Preserve official labels and partitions using `human`/`ai` and `train`/`validation`/`test`; record original IDs/split names in extra columns. Do not randomly manufacture a validation split from the test set.
- `generator=unknown` is allowed only as an explicit limitation, never evidence of generator-disjoint testing. Labeling all unknown generators identically is not proof that they share a family.
- Normalized imports retain metadata columns, never copy or upload audio, and run content/speaker/source leakage checks. Use `--allow-missing` for a declared partial local release. `--limit` selects a deterministic available protocol-order prefix for engineering checks; it is **not a representative sampling strategy**.

```powershell
.\.venv\Scripts\python.exe -m scripts.datasets.import_local --protocol data/official_normalized.csv --audio-root D:\LicensedCorpus --license-file data/REVIEWED_RECORDING_LICENSE.txt --rights-reviewed --allow-missing --limit 100 --output artifacts/datasets/imported.csv
.\.venv\Scripts\python.exe -m research.evaluate audit artifacts/datasets/imported.csv --output artifacts/datasets/audit.json
.\.venv\Scripts\python.exe -m research.evaluate score artifacts/datasets/imported.csv --output artifacts/datasets/scores.csv
```

The importer also supports the **six-field labeled format actually read by the official ASVspoof5 baseline `data_utils.py`**, not every future release layout. Supply `--format asvspoof5-baseline --official-partition train|dev|eval --language en --source-provenance data/source_mapping.csv`. The reviewed source mapping must contain `recording_id,source_id` and optionally `generator`. Original protocol line/recording/speaker/partition are retained; dev maps explicitly to validation, eval to test. A different/unlabeled layout is rejected rather than silently parsed. Use normalized import for the actual documented release's full protocol.

Do not fit policies until there is enough independent validation data, and evaluate frozen policies only on untouched test speakers/sources. Unknown source identity is an external provenance blocker, not a reason to invent IDs.
