# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Published the CNS reference bundle (BICCN Human Brain Cell Atlas v1.0) as release
  `cns-WHB-2023`; the runtime resolves it via `reference_release="WHB-2023"` to run
  the progressive class-gate.

### Changed
- Clarified positioning: rarecell performs rigorous *targeted* isolation of rare
  and hard-to-resolve cell populations and states — not de-novo discovery. README
  rewritten for a precise claim, with explicit scope and a limitations section.
- Core clustering, evidence-scoring, and purification modules reimplemented on
  public libraries (scanpy / scikit-learn / scipy); ~900 net lines removed,
  including unused plotting helpers.

### Fixed
- CNS reference build now completes within a Colab runtime: per-file streaming
  reads in backed mode, downloads are atomic (no truncated cache entries), and a
  highly-variable-gene pre-filter (`--max-genes`, default 5000) bounds CellTypist
  training memory — the full ~58k-gene atlas otherwise OOMs even High-RAM.
