# Bibliography verification scope

Checks made during this research-validation pass. `reference_sources.json` stores the retrieved official-source URL and SHA-256, or the concrete access failure. No benchmark values were imported from these references as VoxGuard measurements.

| Reference | Official source inspected | Verified scope |
|---|---|---|
| AASIST | NAVER authors' `clovaai/aasist` README citation | All eight names/order, exact title, arXiv ID and 2021 preprint year. Existing eight-author list was confirmed correct, not changed speculatively. We cite this verified preprint rather than inventing conference/DOI metadata. |
| RawNet2 | AASIST authors' related-work BibTeX | Six names/order, title, ICASSP 2021 and pages 6369–6373. |
| ASVspoof 2019 database | AASIST authors' related-work BibTeX | Title, listed authors plus `others`, Computer Speech & Language volume 64, article 101114, year 2020. The Markdown DOI is the existing canonical DOI; the DOI endpoint was not independently fetched during this pass. |
| wav2vec 2.0 | Authors' official `facebookresearch/fairseq` wav2vec README | Exact paper title, Baevski lead author, 2020 and arXiv:2006.11477 identity. The primary arXiv abstract request was denied by the proxy. The bibliography conservatively uses lead author `et al.` and arXiv identity rather than claiming a freshly verified full venue/author record. |
| WaveFake | Authors' `RUB-SysSec/WaveFake` README BibTeX | Joel Frank and Lea Schönherr; exact title; NeurIPS Datasets and Benchmarks Track; 2021. The code citation is not a license grant for linked recordings. |

The verification is bibliographic, not a systematic re-review of all papers' experiments. Literature-reported EER or accuracy does not replace a matching locally evaluated baseline. RawNet2 remains unexecuted, and frozen wav2vec representations alone are not a competing spoof classifier. Authors should recheck target-venue citation style, full bibliographic metadata, final published versions and all claims before submission.

Final dataset sprint: the existing official bibliography source records were retained, not represented as a new systematic literature review. Requested dataset card/API endpoints were unreadable; no dataset license or paper-specific experimental claim was inferred from those failures. They are acquisition candidates, not new verified scholarly citations.
