# Changelog

All notable changes to `cc-session-core` are documented here.

## [0.2.0] - 2026-07-25

### Added

- Native, typed parsing for Codex JSONL rollout transcripts.
- A shared discriminated transcript union that automatically routes Claude Code
  and Codex records through one model.
- Codex session auto-detection and normalization into the existing timeline,
  tool-call, usage, cost, export, and investigation views.
- Discovery of active and archived Codex sessions.
- Lossless fallback models for unknown Codex records, events, response items,
  and content variants.
- Synthetic Codex fixtures and coverage for parsing, session loading, tool
  pairing, usage accounting, and tailing.

### Changed

- Expanded Claude Code type coverage for current transcript format drift.
- Updated Pydantic and GitHub Actions dependencies.
- CI now validates Python 3.12, 3.13, and 3.14 with repository-wide Ruff,
  formatting, Pyright, tests, coverage, and package builds.

### Fixed

- Corrected cached-token accounting for normalized Codex usage.
- Created context-map output directories before writing reports.
- Replaced shared mutable model defaults with typed factories.
- Corrected package and publish job gating for release workflows.

[0.2.0]: https://github.com/Magic-Man-us/cc-session-core/compare/v0.1.0...v0.2.0
