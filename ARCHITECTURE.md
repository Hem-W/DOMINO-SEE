# DOMINO-SEE Architecture & Conventions

**Status:** Design document (accepted) — defines the target data model and the refactoring roadmap.
**Scope:** Terminology, xarray data-model conventions, diagnosis of the current codebase, and a phased improvement plan.

DOMINO-SEE builds climate complex networks with an xarray-based pipeline:

```
climate variable ──► eventorize ──► pairwise coupling ──► thresholding ──► network
   (lat, lon, time)    (events)       (eca.py / es.py)      (network.py)     (adjacency)
```

This pipeline design is sound and is kept. What this document changes is **how the
objects flowing through the pipeline are named and represented**, so that the same
conventions work for netCDF today, zarr-chunked large networks tomorrow, and
NetworkX interoperability.

---

## 1. The core representation problem

A spatial network relates **one set of spatial points to itself**: rows and columns
of the adjacency matrix are the *same* nodes. xarray, however, does not allow the
same dimension name to appear twice on one array, so the two sides must carry
different names even though they refer to identical points. The current code solves
this three different ways in three places (see Finding A below), and encodes layer
identity into dimension names (e.g. `drought_locationA`), which forces every
downstream function to parse strings.

The conventions in Section 2 resolve this with one rule set.

## 2. Terminology and data-model conventions

### 2.1 Three rules

1. **Dimension names encode *role only*, never identity.**
   Every network array has the fixed dimensions `node_i × node_j`. Layer names are
   never embedded in dimension names (no `drought_location_i`). Generic code can
   always write `da.sum("node_j")` without discovering names first.

2. **Identity lives in coordinates.**
   *Which point* → integer `node` index with auxiliary coordinates `lat(node)`,
   `lon(node)`. *Which layer* → `layer_i` / `layer_j` coordinates (scalar on a
   single network). Coordinates survive arithmetic, are queryable
   (`da.layer_i.item()`), and serialize to netCDF/zarr. `attrs` are frequently
   dropped by xarray operations and are reserved for non-identity metadata
   (parameters, units, `directed` flag).

3. **One suffix rule for everything per-side.**
   The suffixes `_i` / `_j` apply uniformly to every "sided" entity:
   `node_i`, `lat_i`, `lon_i`, `layer_i` — and nothing else changes between sides.
   Convention for directed networks: edges point **i → j**.

### 2.2 Vocabulary

| Network concept | xarray representation |
|---|---|
| **node** | Integer index dimension `node` with auxiliary coordinates `lat(node)`, `lon(node)`. No `MultiIndex`. Grid-agnostic: a regular lat/lon grid and a Fekete grid produce the same structure. |
| **adjacency rows/cols** | Paired dimensions `node_i` / `node_j` (the mathematical A<sub>ij</sub> convention), auxiliary coordinates `lat_i(node_i)`, `lon_i(node_i)`, `lat_j(node_j)`, `lon_j(node_j)`. |
| **layer** (event type) | At the event stage: scalar coordinate `layer` (set from `event_name`) plus the Dataset variable name. At the network stage: scalar coordinates `layer_i`, `layer_j`. |
| **edge / link** | Boolean adjacency `DataArray` produced by thresholding; `attrs["directed"]` records directedness (ECA networks are directed, ES networks undirected). Generation parameters go to `attrs`. |
| **network** | A `DataArray` with dims `(node_i, node_j)` — weighted (counts, confidence) before thresholding, boolean after. |
| **large-network storage** | Chunked 2-D `(node_i, node_j)` zarr store, one store per layer pair (e.g. `drought__flood.zarr`). Optional sparse form: `pydata/sparse` COO or an edge-list (UGRID-style) encoding. |

**Why an integer `node` index instead of `stack(location=("lat","lon"))`:**
a stacked `MultiIndex` **cannot be written to netCDF or zarr**, which conflicts
directly with the package's storage goals (this is why the tutorial notebooks avoid
stacking and fall into the 4-D code path). The integer index + auxiliary-coordinate
form is serializable, NetworkX-ready (node ids are integers), and identical for
regular and irregular (Fekete) grids. Round-tripping to the original 2-D grid is
provided by `to_node_format` / `from_node_format` helpers (Phase 1), which record
the unstacking information in coordinates/attrs instead of a MultiIndex.

### 2.3 Intra-layer and inter-layer networks

Both kinds exist in DOMINO-SEE and share one representation:

- **Intra-layer**: `layer_i == layer_j` (e.g. drought–drought; typical for ES,
  `directed=False`).
- **Inter-layer**: `layer_i != layer_j` (e.g. drought→flood; typical for ECA,
  directed **i → j**).
- **Multilayer assembly (supra-adjacency)**: a scalar coordinate is the degenerate
  form of a length-1 dimension coordinate, so multiple networks over the *same node
  set* can be assembled with `xr.concat` along `layer_i` / `layer_j`, producing the
  supra-adjacency tensor `(layer_i, layer_j, node_i, node_j)`. Intra-layer networks
  are its diagonal blocks, inter-layer ones the off-diagonal blocks;
  `sel(layer_i="drought", layer_j="flood")` slices one network back out.
  The pairwise (unassembled) form has no shared-node-set constraint — the two sides
  of an inter-layer network may even live on different grids.

Practical notes:

- **Dataset merging**: scalar `layer_i`/`layer_j` coordinates from *different* layer
  pairs conflict inside a single `Dataset` (`MergeError`). Collections therefore use
  either (a) concat-to-supra-adjacency, or (b) one store per layer pair with the
  layer pair in the store name — which is the natural layout for large networks
  anyway.
- **Provenance chain**: `get_event(..., event_name="drought")` writes
  `layer="drought"` as a scalar coordinate (today it only lands in `attrs` and the
  variable name). ECA/ES functions then propagate the two inputs' `layer` coords to
  the output's `layer_i` / `layer_j` automatically.
- **Companion utilities** (Phase 1): `transpose_network()` — swap the i/j sides
  including *all* sided coordinates; `is_intralayer()` — predicate used e.g. for
  symmetric-network optimizations (compute upper triangle only).

### 2.4 Alternatives considered

| Alternative | Why not chosen |
|---|---|
| Keep `locationA` / `locationB` suffixes | Minimal migration, but `location` is not network vocabulary, and A/B reads as two different point sets when they are usually the same set. |
| `source` / `target` dims | Perfect NetworkX alignment for directed nets, but misleading for undirected ES networks. `node_i`/`node_j` is direction-neutral (with the documented i→j convention when direction exists). |
| Layer name inside dimension names (`{layer}_location_i`) | Forces string parsing everywhere (`dims[0].split("_")[0]`), makes dimension names unpredictable, and explodes the dimension namespace in multi-network Datasets. Rejected outright — this is the pattern being removed. |
| Layer identity in `attrs` only | `attrs` are silently dropped by many xarray operations; not robust enough to carry identity. |
| 4-D adjacency `(lat_i, lon_i, lat_j, lon_j)` | Accidentally the current behavior for unstacked input; kept only as an optional *presentation* via `from_node_format`, never the canonical compute/storage form. |

---

## 3. Diagnosis of the current codebase

Findings verified against the code as of this document's commit.

**A. Three coexisting row/column renaming implementations** — the direct source of
the terminology pain:
  1. `dominosee/utils/dims.py:8` `rename_dimensions(suffix="A")`, used by `es.py`
     (`lat → latA`);
  2. `dominosee/eca.py:170-173` inline `rename({var: f"{var}B"})` — duplicated logic;
  3. `dominosee/eca.py:462-523` legacy `get_prec_confidence` / `get_trig_confidence`
     with a third scheme (`lat_locA`, `{layer}_locationA`) that also calls
     `Index.rename(..., inplace=True)` — removed in pandas ≥ 2.0, so these functions
     are **dead code** on modern environments.

**B. Fragile core-dimension selection**: `es.py:318` and `eca.py:178` pick the
spatial core dimension as the *last element of an alphabetically sorted* array
(`np.setdiff1d(dims, ...)[-1]`). For unstacked `(lat, lon)` input only `lon` becomes
a core dim while `lat` falls into the `vectorize` broadcast loop: the output becomes
4-D `(latA, latB, lonA, lonB)`, the numba kernel is re-launched per lat pair, and the
result's dimension order depends on how dimensions happen to sort alphabetically.

**C. No unified node representation**: `utils/dims.py:69` `stack_lonlat` produces a
`MultiIndex`-backed `location` dimension, which cannot be written to netCDF/zarr.

**D. The layer concept is absent from code**: README/ROADMAP advertise multi-layer
networks, but the only related code is commented out (`eventorize.py:300-314`).

**E. `attrs` used as a type system**: `eca.py:164` validates inputs via
`attrs["long_name"] == "Precursor Window"`. Because xarray operations routinely drop
`attrs`, this check breaks on valid data; it should be an explicit parameter or a
check of the data itself.

**F. Scalability bottleneck (conflicts with the zarr goal)**: `es.py:315-327` and
`eca.py:176-183` declare *both* spatial dims as `apply_ufunc` core dims with
`allow_rechunk=True`, so dask must rechunk each spatial dimension into a **single
chunk** before computing — memory blows up exactly when networks get large.
`utils/blocking.py` works around this with a hand-rolled double loop writing many
small netCDF files; the proper replacement is chunk-pair computation with
`to_zarr(region=...)` into one store (Phase 3).

**G. Untested core, aspirational docs**: `eca`, `es`, and `network` — the core of
the package — have **zero tests** (existing tests cover only `eventorize` and
`grid`). `docs/source/user_guide/network_generation.rst` and
`network_analysis.rst` document functions that do not exist
(`create_network`, `create_multilayer_network`, `get_top_nodes`). CI runs docs only
(`.github/workflows/docs.yml`); there is no test workflow.

**H. Engineering details**: version mismatch (`pyproject.toml` `0.0.1-alpha` — not
PEP 440; `__init__.py` `0.0.1`; should be e.g. `0.0.1a0`, single-sourced);
`ipykernel` is a runtime dependency but belongs in dev; `cf-xarray` and `bottleneck`
are declared but unused; ~40% of `eca.py` is commented-out legacy code (already
preserved in git history); source comments mix Chinese and English — standardize on
English for open-source collaboration; `grid/` (~1700 lines) persists grids as
dict + pickle and never produces xarray objects, leaving it disconnected from the
pipeline.

---

## 4. Roadmap

### Phase 1 — Core refactor: adopt the conventions (foundation for everything else)

Files: new `dominosee/conventions.py`; edits to `eca.py`, `es.py`, `eventorize.py`,
`utils/dims.py`, `__init__.py`, `pyproject.toml`.

- `conventions.py`: dimension-name constants; `to_node_format` / `from_node_format`
  (replace `stack_lonlat`, no MultiIndex); one pair-dimension utility replacing the
  three renaming implementations (Finding A); `transpose_network`, `is_intralayer`.
- Delete dead/legacy code (`eca.py:462-523`, commented blocks, obsolete
  `eventorize` remnants).
- Fix core-dim selection (Finding B): operate on the canonical `node` dim, error
  early on un-normalized input instead of silently going 4-D.
- Replace `attrs`-based validation with explicit parameters (Finding E).
- `get_event` writes the `layer` scalar coordinate; ECA/ES propagate
  `layer_i`/`layer_j`.
- Fix version (PEP 440, single source) and prune dependencies (Finding H).

*Acceptance*: all pipeline functions accept/produce `node`-format arrays with
`node_i`/`node_j` outputs; results serialize to netCDF and zarr without
`reset_index` workarounds; no function parses dimension names.

### Phase 2 — Tests and CI

Files: `tests/test_eca.py`, `tests/test_es.py`, `tests/test_network.py`, new
`.github/workflows/test.yml`; rewrite `docs/source/user_guide/*`.

- Brute-force reference tests for the ECA/ES kernels (small numpy cases computed
  the naive way), ES symmetry and ECA directionality (i→j orientation) tests,
  dask-vs-numpy equivalence tests.
- CI matrix across supported Python versions; coverage report.
- Rewrite the user guide to match the real API (Finding G).

*Acceptance*: core kernels covered by reference tests; CI green on the matrix; no
documented-but-nonexistent API remains.

### Phase 3 — Large networks: chunk-pair computation + zarr

Files: new `dominosee/backends/` (or extension of `conventions.py`); retire
`utils/blocking.py`.

- Compute the adjacency block-by-block over pairs of node chunks; write each block
  with `to_zarr(region=...)` into a single `(node_i, node_j)` store.
- Exploit symmetry for intra-layer undirected networks (upper triangle only).
- Optional sparse output (`pydata/sparse` COO / edge list) for thresholded links.

*Acceptance*: a network larger than memory builds end-to-end into one zarr store;
`blocking.py` removed.

### Phase 4 — NetworkX bridge and basic metrics

- `to_networkx(da_link)` → `nx.Graph`/`nx.DiGraph` via `scipy.sparse`, node
  attributes `lat`/`lon` (and `layer` for supra-adjacency input).
- Native xarray implementations for cheap metrics (degree/strength, density) so the
  common cases never need the dense→NetworkX round trip.

*Acceptance*: round-trip node identity preserved; degrees from xarray and NetworkX
agree on test networks.

### Phase 5 — Grid module integration

- `BaseGrid.to_xarray()` producing a `node`-dimension Dataset (lat/lon coords),
  replacing pickle persistence with netCDF/zarr.
- Regridding utilities to map gridded data onto a Fekete grid (existing ROADMAP
  item), emitting `node`-format arrays directly.

*Acceptance*: a Fekete-grid workflow runs through the full pipeline with no special
casing versus regular grids.

---

## Appendix: name migration map

| Current | Target |
|---|---|
| `location` (stacked MultiIndex dim) | `node` (integer index) + `lat(node)`, `lon(node)` coords |
| `locationA` / `locationB`, `latA` / `latB`, … | `node_i` / `node_j`, `lat_i` / `lat_j`, … |
| `{layer}_locationA`, `lat_locA` (legacy) | removed — layer identity moves to `layer_i`/`layer_j` coords |
| `event_name` in `attrs` only | `layer` scalar coordinate (+ `attrs` retained) |
| `rename_dimensions(suffix=...)`, inline renames | single pair-dimension utility in `conventions.py` |
| `stack_lonlat` | `to_node_format` / `from_node_format` |
