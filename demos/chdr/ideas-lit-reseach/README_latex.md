# Combined ChDR literature report

The report combines the setup, analytical single-electron models, FDTD interface treatment and IPPL implementation implications, with 25 numbered literature/software references. Section 4.1 identifies the FEL particle/source capabilities to reuse, the remaining initialization and conservation checks, and the moving-electron-source example in the volume co-edited by Taflove. Section 5 compares measured GHz and optical Cherenkov bands with the current 1D study (60 MeV kinetic, 1 nC, 1 mm rms bunch length), including the comparison figure and experimental references.

## Files required to compile

- `chdr_literature_review.tex`: the canonical consolidated source containing the setup, fixed beam parameters, proof-of-concept scope, literature review and implementation assessment. Its bibliography uses standard BibTeX with natbib and the unsrtnat style.
- `chdr_references.bib`: the reference database; the three undated Meep entries are explicitly marked `n.d.` (no date).
- `output/measurement_bandwidth/measurement_bands_vs_bunch.pdf`: the external comparison figure used in Section 5. Keep this relative path when copying the sources.

The finite-radiator TikZ figure is embedded in the source. The bandwidth plot is an external vector PDF; no custom bibliography style is needed. The original Markdown reviews remain as working material, with the integrated report maintained in the consolidated source.

## Coordinate convention

The report consistently uses Cartesian order `(x,y,z)`, with the electron moving along `+z`. The planar reference has its interface at `x=0`, vacuum at `x>0`, dielectric at `x<0`, and gap `a` measured along `x`. The original setup uses `sigma_z = 3 mm` and 5 nC; Section 5 explicitly uses the current `sigma_z = 1 mm` and 1 nC study. `sigma_x` and `sigma_y` are transverse sizes.

The trajectory, charge/current source, spectral equations, form factor, interface-component descriptions and job-file mapping use this convention. The electron trajectory is `(a,0,vt)`, phase matching fixes `k_z = omega/v`, and the remaining Fourier integral is over `k_y`. Relabelling the axes leaves the numerical beam parameters and internal Cherenkov angle unchanged. The legacy standalone setup is retained separately without edits.

## TeXShop: BibTeX workflow

Use **LaTeX > BibTeX > LaTeX > LaTeX**, with pdfLaTeX as the LaTeX engine. Keep **TeXShop > Settings/Preferences > Engine > BibTeX Engine** set to `bibtex`, as it already is in your setup.

For an automatic build, choose **pdflatexmk** in the engine dropdown beside **Typeset**, then click **Typeset**. This engine is already installed in your TeXShop setup and runs BibTeX and the additional LaTeX passes when needed.

TeXShop's [official release notes](https://pages.uoregon.edu/koch/texshop/version.html) describe the bibliography-engine preference. Its bundled latexmk guide describes automated typesetting. No TeXShop preferences were changed by this repair.

## Command-line build

Run the supplied script from this directory:

    sh build_literature.sh

It runs:

    pdflatex chdr_literature_review.tex
    bibtex chdr_literature_review
    pdflatex chdr_literature_review.tex
    pdflatex chdr_literature_review.tex

The finished PDF is written beside the source and copied to `output/pdf/`. The script does not modify the `.tex` or `.bib` source. Alternatively, run:

    latexmk -pdf -interaction=nonstopmode -halt-on-error chdr_literature_review.tex

In Overleaf, upload `output/pdf/chdr_latex_sources.zip` or the source files plus the figure at its relative path. Select `chdr_literature_review.tex` as the main document with pdfLaTeX.

To regenerate the comparison figure and its wide presentation variant, run
`MPLCONFIGDIR=/tmp/chdr-bandwidth-mpl ~/.venv-h6/bin/python compare_measurement_bands.py`.
The script requires NumPy, pandas and Matplotlib. TeXShop needs only the already-generated PDF to compile the report.

## Bibliography repair and validation

The source previously requested Biber, but TeXShop ran BibTeX, leaving an empty bibliography. The report now uses standard BibTeX with natbib, including numeric citations in order of first citation, DOIs and clickable URLs. That repair preserved the existing title, text, equations, figure, axis labels, numbered bibliography heading and reference metadata. The subsequent coordinate review updated the equations and their explanations to the `+z` convention described above.

The old user-local `biblatex.bst` is no longer involved; this report does not require biblatex or Biber. The temporary compatibility style used during diagnosis was removed. No files in the user's TeX installation or TeXShop preferences were changed.

The final BibTeX and LaTeX build has no warnings or unresolved references. The three Meep software/documentation entries are marked `n.d.` (no date), with their existing access dates retained. All 25 entries appear in the bibliography. The Taflove additions include the 2013 edited volume, the moving-source chapter by Oskooi and Johnson, and the 2005 textbook by Taflove and Hagness. The measurement additions include Tomsk, CLEAR, CESR, ATF2, dielectric-lined-channel and aerogel results. These entries are included in `chdr_references.bib`; the separate research bibliography is not needed to compile the report.

No physics code was changed or simulated. `output/pdf/chdr_latex_sources.zip` is a snapshot of the delivered revision, including these instructions, the build script and the required comparison figure.

## Implementation plan

The edited `chdr_implementation_plan.tex` uses the same `chdr_references.bib`
database with numeric BibTeX citations. Keep these two files together. In
TeXShop, select **pdflatexmk** and Typeset, or run LaTeX > BibTeX > LaTeX > LaTeX.
The source includes the appropriate TeXShop engine hints. The implementation
plan cites 13 works, including MITHRA, Taflove/Hagness, Chew and Ryu. Its PDF is
available beside the source and in `output/pdf/chdr_implementation_plan.pdf`.
