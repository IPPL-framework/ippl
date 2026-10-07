# IPPL cosmology paper draft

This directory contains a journal-neutral starting draft for the IPPL cosmology
paper. Its four numbered sections follow the requested structure:

1. cosmology model;
2. initial conditions;
3. a performance-portable implementation; and
4. validation.

The unnumbered introduction positions the application relative to HACC,
GADGET-4, SWIFT, Abacus, RAMSES, and Nyx. The implementation discussion reuses
the architecture and performance methodology of the published IPPL/ALPINE plasma
paper by paraphrase and citation, not by copying its prose.

Appendix A, `appendix-validation.tex`, records the validation available through
5 October 2026: linear evolution, matched-cosmology Zarija initial conditions,
frozen forces, the analytical pre-crossing pancake, matched nonlinear native
plain-PM FastPM evolution, crossed spatial/translation controls, the completed
Merlin Gaussian CPU study, a separate broadband 1LPT IPPL/FastPM comparison,
and the matched-realization GADGET-2 TreePM comparison. It includes eleven
static result figures, the experiment matrices, observables, acceptance
budgets, retained failures, and report provenance. The 25-frame
dark-matter visualization is retained as a development artifact, not embedded
in the static PDF.

## Build

From this directory:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=tmp/pdfs/build main.tex
cp tmp/pdfs/build/main.pdf output/pdf/ippl-cosmology-paper-draft.pdf
```

The source is intentionally journal neutral. Convert it to the target publisher's
class only after the target venue is selected.

The final draft is `output/pdf/ippl-cosmology-paper-draft.pdf`. Building the
paper uses only the assets in this directory; it does not require the simulation
worktree or rerun any physics. Intermediate build/render files belong under
`tmp/pdfs/`. For visual QA, render the built PDF with `pdftoppm` and inspect all
changed pages before replacing the final draft.

## Figure provenance

The nine PNGs covered by `figures/validation/asset-manifest.json` are byte-identical
copies of the earlier released validation plots, not regenerated or cosmetically
edited results. The additional `zeldovich-fastpm-comparison.png` and
`gadget2-zeldovich-comparison.png` are byte-identical copies of their Merlin
renderings. Their original plot manifests are preserved at
`figures/validation/provenance/zeldovich-fastpm-plot-manifest.json` and
`figures/validation/provenance/gadget2-zeldovich-plot-manifest.json`.
Those manifests record the input report, campaign, plotting source, and output
hashes. Source analysis and particle archives remain on Merlin; only the figures
and small provenance sidecars are copied into the paper. The report inventory
in Appendix A also identifies campaigns without separate figures.

To verify the assets against the local evidence worktree:

```sh
/Users/adelmann/.venv-h6/bin/python -B ../python/ippl-cosmology/scripts/prepare_validation_assets.py \
  --validation-root /Users/adelmann/git/ippl-cosmology-linear --verify-only
```

Omit `--verify-only` to prepare missing copies from the same source. The script
checks exact source reports (or the decompressed bytes of released report
snapshots), preserves original manifests, and refuses conflicting destination
files. It does not overwrite differing assets, rerun simulations, change
tolerances, or manufacture passing results. Absolute source paths describe this
development environment, not a public archival location.

## Evidence boundary

The main-text linear table is the original development snapshot summarized at
`0f2953bde`; its early report lacks the source/executable hash linkage established
for later campaigns. Appendix A distinguishes the later reports and their
recorded provenance. The Merlin Gaussian execution used frozen source
`296fd04c0`; `9506f1e52` is the documentation/audit handoff, not that executable's
source revision.

The supported scope is flat, radiation-free, one-species cosmology with 1LPT
initialization, on an OpenMP CPU backend and tested MPI ranks 1--4. Agreement
with Zarija concerns the matched physical subset and statistics, not identical
random phases, full model coverage, or identical input formats. The new
broadband comparison uses one shared realization and one particle/mesh
resolution; its close IPPL/FastPM agreement is not a convergence result.
Native FastPM was run in ordinary PM mode, not modified FastPM or COLA mode.

Completion is not universal scientific acceptance. The analytical pancake keeps
one strict mass-sum failure. The original nonlinear campaign keeps two timestep
failures, with a separately reported passing refinement extension. The local
spatial study retains 130 failures and qualifies only its first low-mode shell;
the Merlin Gaussian matrix retains 116 particle/mesh-resolution failures and
qualifies no shell across its complete matrix. Close code-to-code agreement is
not proof of continuum convergence. The completed local spatial stage and fresh
Merlin Gaussian stage are separate experiments; the partially completed local
Gaussian runs are not relabeled as a complete campaign.

No nonlinear halo-statistics, GPU, multi-node/exascale performance, or
production-I/O qualification is claimed. This revision adds the completed
four-launch broadband ZB campaign and the one-realization GADGET-2 TreePM
comparison. The latter measures code differences without adding an acceptance
threshold; no existing tolerance was changed.

Red `TODO` markers identify information or evidence still needed before
submission. In particular:

- choose the target journal and author list;
- archive the code, inputs, and machine-readable validation results;
- extend the matched nonlinear PM comparison to higher crossed resolutions,
  independent seeds, and a declared converged science band;
- design a force- and time-step-matched follow-up to isolate TreePM/plain-PM
  differences from integrator differences;
- add 2LPT and quantify initialization transients;
- add CPU/GPU strong and weak scaling on multiple systems;
- add scalable checkpoint/restart and science outputs; and
- replace development-branch references with a public release DOI.
