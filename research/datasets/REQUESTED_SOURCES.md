# Requested-source acquisition and transfer

## Observed status

| User-requested source | Metadata endpoint used | Current evidence |
|---|---|---|
| https://huggingface.co/datasets/mueller91/MLAAD-tiny | `https://huggingface.co/api/datasets/mueller91/MLAAD-tiny?blobs=true` | Proxy tunnel HTTP 403; no readable card, recording license, file catalog or recordings |
| https://zenodo.org/records/14498691 | `https://zenodo.org/api/records/14498691` | Proxy tunnel HTTP 403; no readable release terms or file catalog |
| https://huggingface.co/datasets/mueller91/In-The-Wild | `https://huggingface.co/api/datasets/mueller91/In-The-Wild?blobs=true` | Proxy tunnel HTTP 403; no readable card, recording license, file catalog or recordings |

These are the requested URLs, not independently established publisher/recording-rights claims. The earlier bounded checks failed; the resumed task did not retry the same blocked requests. **Dataset licenses, redistribution permissions, exact file sizes, partition layouts and available metadata remain unverified here. Zero new recordings were acquired.** Do not substitute a source-code license for recording rights or assume a similarly named mirror has the same rights. A representative benchmark, calibration policy and real hybrid comparison remain unexecuted.

The commands below run on a computer with an independently permitted connection. They do not change this environment's network policy. Use the site's normal authorized authentication if its release requires it; never put tokens in commands, catalogs, inventories or chat.

## 1. Inspect metadata without downloading audio

From a checkout containing these helpers, PowerShell:

```powershell
New-Item -ItemType Directory -Force artifacts\catalogs | Out-Null
python -m scripts.datasets.source_catalog --source mlaad-tiny --output artifacts\catalogs\mlaad-tiny.json
python -m scripts.datasets.source_catalog --source in-the-wild --output artifacts\catalogs\in-the-wild.json
python -m scripts.datasets.source_catalog --source zenodo-14498691 --output artifacts\catalogs\zenodo-14498691.json
```

Each output is exclusive; use a fresh filename for a new inspection. Unavailable metadata exits with status 2 and records the access failure. Successful metadata is **not permission**. Inspect the source card, release terms and links to the responsible authors; retain the actual recording/dataset terms and permitted-use conditions in `data\REVIEWED_RECORDING_LICENSE.txt`. Record the authoritative license URL separately. Confirm local academic use and any transfer/redistribution restrictions. If rights cannot be verified, stop acquisition. If audio is synthetic-only, that release alone cannot support two-class discrimination metrics.

## 2. Select bounded files from an immutable HF revision

Install the optional official client on the permitted acquisition computer:

```powershell
python -m pip install "huggingface_hub>=0.25,<2"
```

Create `data\selected-files.txt`, **one exact `rfilename` from the inspected catalog per line**. Do not guess filenames, use wildcards, or select a whole repository. Include the necessary official metadata/protocol files as well as a deliberately bounded audio/shard subset. Unknown file sizes, missing revision SHA, more than 32 names, duplicate names, unsafe paths or a selection exceeding the budget are rejected. The helper pins the inspected 40-character revision; it never calls unbounded `snapshot_download`.

For example, this command limits the *inspected selected-file total* to 256 MiB; it is not an assertion that such a subset exists:

```powershell
python -m scripts.datasets.hf_subset --catalog artifacts\catalogs\mlaad-tiny.json --files data\selected-files.txt --output artifacts\acquisition\mlaad-tiny --license-file data\REVIEWED_RECORDING_LICENSE.txt --max-bytes 268435456 --rights-reviewed
```

For In-The-Wild, use its catalog and a new output directory. SDK downloads use the authorized official route and support their normal local retry/resume behavior. The helper checks available storage, downloaded sizes and LFS SHA-256 where supplied, then records observed digests. A digest merely observed after HTTPS transfer is explicitly distinguished from an author/LFS digest. The byte budget is a metadata preflight; post-download checks fail if the server content differs. No release has been fetched or proven to meet the budget here.

For Zenodo, inspect `files` in the returned release metadata and use the existing bounded HTTPS helper with the **actual official file URL, expected digest and byte size**, after reviewing terms:

```powershell
python -m scripts.datasets.download --help
```

There is no verified filename/checksum to insert for record 14498691 here, so none is invented. If a release provides no independently trusted expected SHA-256, record a separately trusted acquisition digest and its provenance before using that helper. Private/signed/query-bearing URLs are intentionally unsupported; use the platform's normal browser/client and the transfer inventory below rather than altering authentication URLs. Do not extract an enormous archive to obtain a few samples unless selective access and storage have been justified.

## 3. Prepare a reviewed portable bundle

Preserve downloaded files, `acquisition.json`, retained recording rights and official protocol metadata. Inspect the actual release format before converting anything. This workflow does not guess audio/label fields in an unreadable Parquet shard or undocumented archive. Release-specific conversion remains blocked until its real layout is available.

Once justified, prepare `protocol.csv` using the existing normalized schema:

`audio_path,label,speaker_id,source_id,generator,split,language`

Audio paths must be relative to the bundle's audio root. Preserve original recording IDs and official partitions in extra columns; labels must explicitly map to `human`/`ai` and partitions to `train`/`validation`/`test`. Do not infer source groups or speakers from filenames. Unknown speaker/source provenance blocks leakage-free evaluation; unknown synthesis identity blocks unseen-generator claims. Keep original recordings and all derived variants in one partition. Never manufacture a validation split from final test data just to enable verdicts.

Create an explicit `data\transfer-files.txt` listing **only the intended paths relative to the bundle root**, such as the selected recordings, original protocols, reviewed rights, normalized manifest and acquisition record. The list is release-specific and must be supplied by the custodian. Do not include credentials, SDK caches or unrelated recordings.

```powershell
python -m scripts.datasets.transfer seal --root D:\AuthorizedBundle --files data\transfer-files.txt --output artifacts\transfer-inventory.json
```

The command prints `inventory_sha256=...`. Retain that digest via a trusted channel separate from the transferred inventory. Transfer the bundle and inventory using a permitted method, such as approved removable storage. Observed transfer hashes establish integrity against that trusted inventory, **not authorship or permission**.

## 4. Verify and import on the evaluation computer

Replace the digest below with the actual value retained above:

```powershell
python -m scripts.datasets.transfer verify --root D:\AuthorizedBundle --inventory D:\transfer-inventory.json --inventory-sha256 TRUSTED_INVENTORY_SHA256
python -m scripts.datasets.import_local --protocol D:\AuthorizedBundle\protocol.csv --audio-root D:\AuthorizedBundle\files --license-file D:\AuthorizedBundle\REVIEWED_RECORDING_LICENSE.txt --rights-reviewed --allow-missing --output artifacts\datasets\imported.csv
```

The importer never copies recordings into Git and never invents identities. `--allow-missing` explicitly permits a partial local release, with missing counts recorded. Optional `--limit` is a protocol-order engineering prefix, **not representative sampling**; any research subset requires a separately frozen selection rule and supportable population claim. The ASVspoof5-specific adapter is documented in `VALIDATION_ACCESS.md`; it does not establish that these three different requested releases use that layout.

## 5. Freeze, evaluate and preserve abstention

After rights, both classes and original split/group provenance are established:

```powershell
python -m research.validation.run artifacts\datasets\imported.csv --output artifacts\validation\independent-001 --dataset "Verified release, revision and subset rule" --conditions "Verified language, channel and original protocol" --fit-policy --robustness
```

Use a fresh run directory. Only request generator-held-out analysis when generator family metadata is verified. Calibration requires sufficient independent validation data; an absent or inadequate partition must leave production **Uncertain**. The executable runner records provenance and refusal evidence instead of installing arbitrary thresholds. The commands are runnable acquisition/evaluation mechanics, not evidence that representative data has become available.
