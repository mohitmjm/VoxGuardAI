# Dataset rights and acquisition record

This project distinguishes recording rights from model/code licenses. Do not assume an MIT code repository licenses linked voice recordings. No large speech dataset or voice recording is committed to Git.

## Reproducible convenience pilot actually acquired

Run from the project root:

```sh
python scripts/fetch_evaluation_samples.py
python -m research.evaluate audit artifacts/pilot/manifest.csv --output artifacts/pilot/audit.json
python -m research.evaluate score artifacts/pilot/manifest.csv --output artifacts/pilot/scores.csv
python -m experiments.robustness artifacts/pilot/manifest.csv --output artifacts/pilot/robustness.csv
```

The fetcher downloads three unmodified Ogg excerpts from the LibriSpeech SLR12 corpus, as distributed in the [librosa-data repository](https://github.com/librosa/librosa-data/tree/38f4b06556fa0ff1acda5e677d8ba05d1bc0fff0/audio) at immutable commit `38f4b06556fa0ff1acda5e677d8ba05d1bc0fff0`. It fetches and verifies the SHA-256 of each recording's attribution TOML **before** fetching its audio. All three TOML files explicitly state **CC-BY-4.0**. Full attribution is retained in the output directory and `provenance.json`; derivatives must preserve attribution and indicate modifications. Consult the actual [CC-BY-4.0 terms](https://creativecommons.org/licenses/by/4.0/) before redistribution.

| Excerpt ID | Work / author | Reader | Rights inspected |
|---|---|---|---|
| 5703-47212-0000 | The Ashiel Mystery / Marion Bryce | Garth Comira | [Per-recording TOML](https://github.com/librosa/librosa-data/blob/38f4b06556fa0ff1acda5e677d8ba05d1bc0fff0/audio/5703-47212-0000.toml) |
| 3436-172162-0000 | The Age of Chivalry / Thomas Bulfinch | Anders Lankford | [Per-recording TOML](https://github.com/librosa/librosa-data/blob/38f4b06556fa0ff1acda5e677d8ba05d1bc0fff0/audio/3436-172162-0000.toml) |
| 198-209-0000 | Sense and Sensibility / Jane Austen | Heather Barnett | [Per-recording TOML](https://github.com/librosa/librosa-data/blob/38f4b06556fa0ff1acda5e677d8ba05d1bc0fff0/audio/198-209-0000.toml) |

Three additional recordings use original project-authored text and the installed FFmpeg `flite` filter, with voices `slt`, `awb`, and `rms`. [Flite COPYING](https://github.com/festvox/flite/blob/master/COPYING) was inspected: its software is freely usable under the documented CMU permissive license, with stated exceptions. The installed FFmpeg version and original texts are retained in provenance. This generator is **classical, non-neural TTS**; labeling its output synthetic does not establish modern AI/voice-cloning capability. Windows FFmpeg distributions may omit libflite; the fetch script reports an error rather than fabricating replacement samples.

All six recordings are marked **test**. There are no training or validation records and no threshold is fit to them. They constitute an intentionally tiny convenience pilot, not ASVspoof evaluation. Different recording/source domains confound this comparison. Rank separation on six examples cannot establish accuracy, probability calibration, unseen-generator detection, or publishable robustness. Synthetic voices derive from human voice inventories, so speaker provenance must be re-audited before combining these with any corpus containing the same voice donors.

## Larger candidate datasets inspected for availability

| Candidate | Authoritative acquisition destination | Actual status in this environment |
|---|---|---|
| ASVspoof 2019 LA | https://www.asvspoof.org/ and the official release links | ASVspoof site requests denied by network proxy (HTTP tunnel 403). Dataset license not retrieved; no ASVspoof recordings downloaded. AASIST's MIT model/code license is not a dataset license. |
| ASVspoof 5 | https://www.asvspoof.org/index2024.html | Site denied by network proxy. Current release terms and partitions require inspection before acquisition; no inferred license. |
| WaveFake | https://zenodo.org/records/5642694 | [Authors' code README](https://github.com/RUB-SysSec/WaveFake) inspected and links this record. Zenodo API denied by proxy. Dataset license not verified; do not infer it from repository code. |
| In-the-Wild | https://deepfake-demo.aisec.fraunhofer.de/in_the_wild | Authoritative host denied by proxy. Distribution/recording rights not verified here; no files downloaded. |
| Indic human speech candidates | https://www.openslr.org/ (e.g. resource 103 must be checked for its exact languages) | OpenSLR denied by proxy. No Hindi/Marathi human or labeled synthetic pairs acquired; language and release-specific terms must be verified independently. |

Required domains for future authorized acquisition: `www.asvspoof.org`, `zenodo.org`, `deepfake-demo.aisec.fraunhofer.de`, and `www.openslr.org`. A domain addition alone does not grant a recording license or validate content. Network settings were not broadened automatically. Large-dataset downloads should follow explicit release/protocol review and available storage checks.

## Corpus ingestion requirements

Use the existing manifest schema: `audio_path,label,speaker_id,source_id,generator,split,language`. Keep source recordings and all derivatives in one partition; namespace IDs globally and preserve original speaker and generator family information. Audit manifests before inference. For a generator-held-out study, exclude held-out generator families from both training **and validation**, not just training. ASVspoof pretrained overlap must be checked against the checkpoint's original training set. Store rights, license version, acquisition URL, record count, hashes, and consent restrictions alongside a frozen manifest.

## Current validation acquisition tools

See [VALIDATION_ACCESS.md](VALIDATION_ACCESS.md) for the new authoritative access/rights investigation, bounded resume downloader, strict manual/partial imports and ASVspoof5 baseline protocol adapter. No new representative modern-synthesis corpus was acquired. Tests: `python -m pytest research/datasets/test_acquisition.py -q`.
