# Combined ChDR literature report

The report combines the setup, analytical single-electron models, FDTD interface treatment and IPPL implementation implications, with 16 numbered literature/software references. Section 4.1 identifies the FEL particle/source capabilities to reuse, the remaining initialization and conservation checks, and the moving-electron-source example in the volume co-edited by Taflove.

## Files required to compile

- `chdr_literature_review.tex`: the author's latest edited report, with its bibliography configuration converted to standard BibTeX using natbib and the unsrtnat style.
- `chdr_references.bib`: the reference database; the three undated Meep entries are explicitly marked `n.d.` (no date).

The finite-radiator TikZ figure is embedded in the source; no external images or custom bibliography styles are needed. The original Markdown reviews and `../chdr_setup.tex` are retained separately.

## Coordinate convention

The report consistently uses Cartesian order `(x,y,z)`, with the electron moving along `+z`. The planar reference has its interface at `x=0`, vacuum at `x>0`, dielectric at `x<0`, and gap `a` measured along `x`. The longitudinal bunch length is `sigma_z = 3 mm`; `sigma_x` and `sigma_y` are transverse sizes.

The trajectory, charge/current source, spectral equations, form factor, interface-component descriptions and job-file mapping use this convention. The electron trajectory is `(a,0,vt)`, phase matching fixes `k_z = omega/v`, and the remaining Fourier integral is over `k_y`. Relabelling the axes leaves the numerical beam parameters and internal Cherenkov angle unchanged. The original setup draft is retained separately without edits.

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

In Overleaf, upload the two source files and select `chdr_literature_review.tex` as the main document with pdfLaTeX.

## Bibliography repair and validation

The source previously requested Biber, but TeXShop ran BibTeX, leaving an empty bibliography. The report now uses standard BibTeX with natbib, including numeric citations in order of first citation, DOIs and clickable URLs. That repair preserved the existing title, text, equations, figure, axis labels, numbered bibliography heading and reference metadata. The subsequent coordinate review updated the equations and their explanations to the `+z` convention described above.

The old user-local `biblatex.bst` is no longer involved; this report does not require biblatex or Biber. The temporary compatibility style used during diagnosis was removed. No files in the user's TeX installation or TeXShop preferences were changed.

The final BibTeX and LaTeX build has no warnings or unresolved references. The three Meep software/documentation entries are marked `n.d.` (no date), with their existing access dates retained. All 16 entries appear in the bibliography. The Taflove additions include the 2013 edited volume, the moving-source chapter by Oskooi and Johnson, and the 2005 textbook by Taflove and Hagness. These entries are included in `chdr_references.bib`; the separate research bibliography is not needed to compile the report.

No physics code was changed or simulated. `output/pdf/chdr_latex_sources.zip` is a snapshot of the delivered revision, including these instructions and the build script.
