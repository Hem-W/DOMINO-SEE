# DOMINO-SEE Roadmap

This document outlines the planned future development for the DOMINO-SEE package.

The architectural direction, terminology conventions, and the detailed phased plan
are defined in [ARCHITECTURE.md](ARCHITECTURE.md). The phases below track that
document; finer-grained items from the previous roadmap are kept under the phase
they belong to.

## Phase 0 — Architecture & conventions (done)

- [x] Define terminology and conventions: map network terminology to xarray terminology
      (`node`, `node_i`/`node_j`, `layer_i`/`layer_j` — see ARCHITECTURE.md §2)
- [x] Add basic tests to examine the correctness of calculations (eventorize, grid)
- [x] Add basic documentation
- [x] Implement BaseGrid and FeketeGrid classes
- [x] Create contribution guidelines
- [x] Add examples directory with sample scripts

## Phase 1 — Core refactor: adopt the conventions

- [ ] Add `dominosee/conventions.py`: dimension-name constants,
      `to_node_format`/`from_node_format` (replaces `stack_lonlat`, no MultiIndex),
      one pair-dimension utility (replaces the three renaming implementations),
      `transpose_network`, `is_intralayer`
- [ ] Delete dead/legacy code (`eca.py` legacy confidence functions and commented blocks)
- [ ] Fix fragile spatial-core-dimension selection in `eca.py`/`es.py`
- [ ] Replace `attrs`-based input validation with explicit parameters
- [ ] Write `layer` as a scalar coordinate in `get_event`; propagate `layer_i`/`layer_j`
      through ECA/ES outputs
- [ ] Fix version numbering (PEP 440, single-sourced) and prune dependencies
- [ ] Support unsymmetric event synchronizations (ES)

## Phase 2 — Tests, CI, and truthful docs

- [ ] Brute-force reference tests for ECA/ES kernels; ES symmetry and ECA
      directionality tests; dask-vs-numpy equivalence tests
- [ ] Add a GitHub Actions test workflow (Python version matrix, coverage)
- [ ] Rewrite `docs/user_guide` pages that document non-existent APIs
- [ ] Add advanced tests to examine the bugs in various working environments

## Phase 3 — Large networks: chunk-pair computation + zarr

- [ ] Chunk-pair blockwise computation writing into a single `(node_i, node_j)` zarr
      store via `to_zarr(region=...)`; retire `dominosee/utils/blocking.py`
- [ ] Exploit symmetry for intra-layer undirected networks (upper triangle only)
- [ ] Optional sparse representation (pydata/sparse COO or edge list) for link matrices

## Phase 4 — Network analysis & NetworkX interoperability

- [ ] Native xarray implementations of basic metrics (degree/strength, density)
- [ ] `to_networkx()` conversion via scipy.sparse with node attributes (lat/lon, layer)
- [ ] Multilayer export from supra-adjacency form
- [ ] Add regional bundle analysis
- [ ] Add visualization capabilities (candidate backends to be evaluated)

## Phase 5 — Grid integration & event tooling

- [ ] `BaseGrid.to_xarray()` producing `node`-format Datasets; replace pickle persistence
- [ ] Complete and test grid implementation
- [ ] Add regrid utilities for remapping to Fekete grid
- [ ] Introduce `xclim` for time period subsetting (specific months/days)
- [ ] Introduce `xclim` for event sub-selection
- [ ] Add VLType for event timing extraction if `xarray` supports VLType

## Community Features

- [ ] Purge ROADMAP out to GitHub Discussion

## How to Contribute

If you're interested in contributing to any of these initiatives, please submit a
pull request or open an issue to discuss your ideas.
