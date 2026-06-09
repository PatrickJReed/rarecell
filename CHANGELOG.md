# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Clarified positioning: rarecell performs rigorous *targeted* isolation of rare
  and hard-to-resolve cell populations and states — not de-novo discovery. README
  rewritten for a precise claim, with explicit scope and a limitations section.
- Core clustering, evidence-scoring, and purification modules reimplemented on
  public libraries (scanpy / scikit-learn / scipy); ~900 net lines removed,
  including unused plotting helpers.
