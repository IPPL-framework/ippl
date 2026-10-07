# Paper task state

## Completed update: matched GADGET-2 comparison (2026-10-05)

Added the completed GT experiment to the existing Validation appendix. The new
subsection documents the shared saved $z_i=99$ 1LPT realization, the small
GADGET Format-1 conversion roundoff, the TreePM/plain-PM force and timestep
differences, the common CIC power estimator, and exactly what “Fourier shell”
and its plotted units mean. It includes the final $z=0$ three-code power plot,
selected offsets and provenance. Added the GADGET-2 source citation
(Springel 2005, DOI 10.1111/j.1365-2966.2005.09655.x). The text explicitly
limits this to a single-realization code comparison: the large high-$k$
TreePM/plain-PM offset is not a convergence result or an acceptance failure,
and the unmatched force/timestep choices prevent attributing it solely to
TreePM.

The authoritative campaign, analyzer report and raw numerical outputs remain
on Merlin; no solver was rerun for this manuscript edit. The PNG and its
provenance sidecar are the only copied GADGET comparison assets. The common
initial-condition fixture is the same saved CSV used by the earlier IPPL and
FastPM comparison, not a separately randomized realization.

`main.tex` was not edited. Its SHA-256 remains
`7e537b1970d20c5cdc3e548250ace380b94cee1f26a873944f3659ba998f5512`.

### Files and evidence

- `appendix-validation.tex`: added campaign GT to the summary/provenance and
  added subsection `app:gadget2-zeldovich`, its comparison table and figure.
- `references.bib`: added the primary GADGET-2 paper citation.
- `README.md`: updated the as-of date, figure count, comparison scope and
  remaining force-/step-matched work.
- `figures/validation/gadget2-zeldovich-comparison.png`: SHA-256
  `5890ad8f315e9cf2fe0979aa7ac02755a49562a4be338b1a1d74e8747b4948f6`.
- `figures/validation/provenance/gadget2-zeldovich-plot-manifest.json`: SHA-256
  `46274a10e89e0a51d421912c1505b54fa87e09a7b2c130bf4da12193a8418f95`.
- Merlin analysis report SHA-256: `8ad14e25a766665ce3f8c347c250637a2692fd54e06f1a0ea8ac8d3f9b5605a9`;
  campaign SHA-256: `9d7c0fa83907641272c2a7fa677857e475e897cf858349ea8c7a50fbc35cf80d`;
  plot input snapshot SHA-256: `4f10e99fc39146fb5dd17aefc426d07f229df515a4a2dd458eef8345a7df74b5`.

### Build and verification

Rebuilt with `latexmk -pdf -interaction=nonstopmode -halt-on-error
-outdir=tmp/pdfs/gadget2-appendix main.tex`; final output is 36 pages. The
GADGET reference resolved, and the final LaTeX log has no unresolved
references or overfull boxes. Rendered pages 10--11 and 30--34 were visually
checked, including the citation, comparison table, plot, and provenance. The
final PDF at `output/pdf/ippl-cosmology-paper-draft.pdf` has SHA-256
`52effe2fb0c07c31163c80ce8ec0ff74aaa9efbb8eaf71d024f24651a159294a`.
The prior draft PDF was preserved at `tmp/pdfs/pre-gadget2-appendix.pdf`. No
commit or push was made.

## Completed update: broadband Zel'dovich/FastPM benchmark (2026-10-04)

Added the completed Merlin CPU benchmark to Appendix A as campaign ZB. The new
subsection describes the shared 1LPT realization, pure-BBKS model, native
plain-PM FastPM/IPPL setup, run integrity, final phase-space and power-spectrum
differences, and the limits of the comparison. It explicitly distinguishes
the shot-noise-subtracted CIC spectrum used by ZB from the direct-mode spectra
in earlier campaigns. The paper now includes the benchmark figure and its
separate original plot manifest; raw particle and spectrum archives remain on
Merlin. README evidence limits now identify the single-realization, single-
resolution scope and preserve the original nine-asset manifest unchanged.

`main.tex` was not edited. Its SHA-256 before and after this update is
`7e537b1970d20c5cdc3e548250ace380b94cee1f26a873944f3659ba998f5512`.
No simulation source, tolerance, or campaign data was changed or rerun. No
commit or push was made.

### Update files and evidence

- `appendix-validation.tex`: added ZB campaign entry, methods, metrics, table,
  plot and report provenance; clarified which power estimator is used by each
  study.
- `README.md`: added the ZB result, plot/manifest provenance, and scope limits.
- `figures/validation/zeldovich-fastpm-comparison.png`: byte-identical to the
  rendered Merlin plot; SHA-256 `c872f82e846c16825134d2193dbc74692816b4ff2151f13b1543986db20d311c`.
- `figures/validation/provenance/zeldovich-fastpm-plot-manifest.json`: preserved
  plot provenance; SHA-256
  `01b8a458c4b097aec340d73c08a3a3dc8034b9e7bbb1fe3c733740c7a2073114`.
- Authoritative Merlin campaign and analysis reports:
  `/data/user/adelmann/cosmology-zeldovich-broadband-20261004/campaign.json`
  and `analysis/results.json`; their recorded SHA-256 values are respectively
  `340b497e334d50767bf83020185650e328ec644db498e3b0b2988d7d63ce0129` and
  `b345da2fbbf22e9a57091a06352be15da72d6296809502a571fdc130453f6122`.

### Verification

The final manuscript was rebuilt to 33 pages and changed pages 10--11 and
28--30 were rendered and visually inspected. These include the campaign
summary, updated spectrum conventions, new subsection, table, figure, and
adjoining provenance section. The build log has no overfull boxes or unresolved
references. All nine previously curated PNG hashes and their five saved source
manifest hashes match `asset-manifest.json`; the ZB PNG hash also matches its
separate original plot manifest. Re-running the source-based
`prepare_validation_assets.py --verify-only` is currently blocked because the
preserved evidence worktree lacks
`build_openmp/demos/cosmology/zarija-plots-final/plot_manifest.json`; this is a
missing source input, not a mismatch in the already copied paper assets.
PDF path: `output/pdf/ippl-cosmology-paper-draft.pdf`. The new result does not
turn the comparison into a convergence, GADGET-2, GPU, or exascale validation.

## Completed update: detailed Validation appendix (2026-10-04)

User confirmed that the requested destination is this existing manuscript,
`papers/ippl-cosmology`, not a new `papers/cosmology` directory. Added an appendix
named Validation with the completed linear, matched-Zarija, frozen-force,
analytical-pancake, matched nonlinear PM, spatial-control and Merlin Gaussian
evidence, including plots. Preserved the manuscript's structure and unrelated
dirty worktree changes. No new simulations or changes to acceptance budgets.

Completed: independently audited the saved reports and figure manifests; wrote
a separate appendix source; narrowly updated stale main-text scope claims;
copied nine verified figure assets with provenance; rebuilt the paper and
visually inspected all pages, then rechecked the affected pages after layout
polish. The PDF skill guided the render/inspect workflow. Completed execution
is distinguished from scientific qualification, and the separate Mac spatial
and Merlin Gaussian studies are not presented as one machine campaign.

Evidence source: `/Users/adelmann/git/ippl-cosmology-linear` at handoff commit
9506f1e52, with immutable per-campaign report hashes. Merlin ran frozen source
296fd04c0; no A100 runtime or exascale performance evidence exists. The complete
Gaussian study retains 116 resolution failures and no qualified full-matrix
shell. Original manuscript is an untracked user-owned directory; do not commit
or push it as an incidental part of this documentation request.

### Changed files and verification

- `appendix-validation.tex`: Appendix A, nine subsections, nine figures, four
  tables and four equations; final appendix spans pages 10--28.
- `main.tex`: appendix integration, landscape figure macro and table packages;
  abstract, scope, validation, roadmap and availability updated consistently.
  Corrected the units terminology from dimensionless to rescaled potential.
- `figures/validation/`: nine byte-identical released PNGs, five preserved
  manifests, and `asset-manifest.json`; 3,195,329 bytes of PNGs. Manifest SHA-256:
  `09f3c398b916abab3f25e6bd649809ae69d050de03691cde8a222676297bb03b`.
- `scripts/prepare_validation_assets.py`: reproducible, stdlib-only asset
  preparation/verification; preflight conflict checks prevent partial overwrite.
- `README.md`: build, figure verification, provenance and evidence limits.
- `output/pdf/ippl-cosmology-paper-draft.pdf`: final 30-page manuscript.

Checks completed: `latexmk` final exit 0, resolved citations/references and no
overfull boxes or LaTeX warnings; all pages visually checked with Poppler;
revised Gaussian/provenance pages re-rendered and checked; asset verification
passes for 15 files; source whitespace/conflict-marker check and `git diff
--check` pass. The latter does not cover the untracked manuscript, so its source
files were checked separately. Asset-script idempotence, missing-output
verify-only behavior, hash/ambiguity rejection and conflict preflight were also
tested during preparation. Existing scientific results were audited, not rerun.

All review corrections are incorporated: Zarija MPI runs are statistical (only
IPPL rank comparisons share phases); the nine scalar exclusions are one DC
and eight table diagnostics; IC MPI RMS is componentwise; nonlinear mode pairs
are unique, not statistically independent; PCS extraction applies only to G;
crossed spatial failures range 1.11--1.39%. Early L provenance limitations remain
explicit. Saved failures are never recast as passing criteria.

The original nine-page PDF is preserved at `tmp/pdfs/pre-validation-appendix.pdf`.
No simulation source, acceptance tolerance, bibliography, or unrelated worktree
change was modified; no commit or push was made. Large numerical archives remain
outside this manuscript. Author, venue, funding and performance TODOs remain
intentionally visible in the draft.

## Goal

Develop a source-grounded paper in parallel with the IPPL cosmology code, using
the requested structure: cosmology model, initial conditions,
performance-portable implementation, and validation. Reuse relevant ideas from
the published IPPL plasma mini-app paper and determine the role of an IPPL-based
cosmology code in the research landscape.

## Current draft

- `main.tex`: journal-neutral manuscript with the four requested numbered
  sections, an abstract, research positioning, roadmap, and evidence limits.
- `appendix-validation.tex`: detailed validation through 4 October 2026, with
  numerical definitions, criteria, results, retained failures and provenance.
- `references.bib`: primary literature for IPPL/ALPINE, Kokkos, heFFTe, HACC,
  GADGET-4, SWIFT, AbacusSummit, RAMSES, Nyx, and the large-scale simulation
  review.
- `README.md`: build instructions and the boundary between measured and planned
  claims.

## Design decisions

- Position the current code as a performance-portable PM research and validation
  platform, not a production competitor to HACC, GADGET-4, SWIFT, or Abacus.
- Keep the original linear snapshot `0f2953bde` distinct from later campaigns,
  frozen execution sources and documentation/audit commits.
- Reuse IPPL plasma-paper architecture and methodology by paraphrase and citation.
- Mark missing evidence as red `TODO` text rather than inventing results.
- Keep the manuscript journal neutral until a venue is selected.

## Sources checked

- S. Muralikrishnan et al., SIAM PP24, DOI
  `10.1137/1.9781611977967.3`.
- Primary papers for HACC, GADGET-4, SWIFT, AbacusSummit, RAMSES, Nyx, Kokkos,
  and heFFTe; citations are in `references.bib`.
- Repository code under `demos/cosmology` and development history/state on
  branch `codex/cosmology-linear`.

## Next step

The requested appendix and checked 30-page PDF are ready for author review.
Next, select the publication scope/venue and prepare a durable code-and-evidence
archive. Subsequent scientific work should resolve the crossed particle/mesh
sensitivities and add independent seeds, 2LPT/transient and energy checks before
broader nonlinear claims; GPU/multi-node correctness and scaling still require
their own campaigns. Do not silently relax the existing budgets or backfill
missing runtime evidence.
