PYTHON ?= $(HOME)/.venv-h6/bin/python
LATEXMK ?= latexmk
TEX := chdr_1d_presentation.tex
PDF := chdr_1d_presentation.pdf
BAND_COMPARISON := ../../ideas-lit-reseach/compare_measurement_bands.py
BAND_FIGURE := ../../ideas-lit-reseach/output/measurement_bandwidth/measurement_bands_vs_bunch_slides.pdf

.PHONY: all figures literature-figure pdf clean

all: pdf

figures: literature-figure
	$(PYTHON) make_figures.py

literature-figure:
	MPLCONFIGDIR=$(CURDIR)/.matplotlib XDG_CACHE_HOME=$(CURDIR)/.cache $(PYTHON) $(BAND_COMPARISON)
	mkdir -p figures
	cp $(BAND_FIGURE) figures/measurement_bands_vs_bunch.pdf

pdf: figures $(TEX)
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error $(TEX)

clean:
	$(LATEXMK) -C $(TEX)
