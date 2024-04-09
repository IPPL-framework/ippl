# Makefile for running Doxygen with a Doxyfile

# Define variables
DOXYGEN = doxygen
DOXYFILE = Doxyfile

# Default target
.PHONY: all
all: doxygen

# Target to run Doxygen
.PHONY: doxygen
doxygen:
	$(DOXYGEN) $(DOXYFILE)

# Clean target
.PHONY: clean
clean:
	rm -rf html latex