#!/bin/sh
# Build the current author-edited source with the requested BibTeX backend.
set -eu
cd -- "$(dirname -- "$0")"

# MacTeX's tools may be missing from a terminal or editor's PATH.
if ! command -v pdflatex >/dev/null 2>&1 || ! command -v bibtex >/dev/null 2>&1; then
    PATH="${PATH}:/Library/TeX/texbin"
    export PATH
fi
for tool in pdflatex bibtex; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        printf 'Required tool missing: %s. Install TeX Live or MiKTeX.\n' "$tool" >&2
        exit 1
    fi
done

pdflatex -interaction=nonstopmode -halt-on-error -synctex=1 chdr_literature_review.tex
bibtex chdr_literature_review
pdflatex -interaction=nonstopmode -halt-on-error -synctex=1 chdr_literature_review.tex
pdflatex -interaction=nonstopmode -halt-on-error -synctex=1 chdr_literature_review.tex

mkdir -p output/pdf
cp chdr_literature_review.pdf output/pdf/chdr_literature_review.pdf
printf 'Built chdr_literature_review.pdf and output/pdf/chdr_literature_review.pdf\n'
