# Build the SpecKit manuscript.
#
#   make            build main.pdf
#   make clean      remove TeX intermediates, keep main.pdf
#   make distclean  remove intermediates and main.pdf
#
# Run from this directory, or as `make -C docs/paper` from the repository root.
# latexmk is invoked with -cd, so the same Makefile also works as
# `make -f docs/paper/Makefile`.

MAIN := main
PAPER_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

LATEXMK ?= latexmk
# bibtex_use=1.5 runs BibTeX when bibliography.bib changes, and lets
# `make clean` delete the generated .bbl. The latexmk default keeps it.
LATEXMK_COMMON = -e '$$bibtex_use=1.5'
LATEXMKFLAGS := -pdf -interaction=nonstopmode -halt-on-error -file-line-error -synctex=1

# MacTeX installs its binaries in /Library/TeX/texbin, which is not always on PATH.
ifeq ($(shell command -v $(LATEXMK) >/dev/null 2>&1 && echo yes),)
  ifneq ($(wildcard /Library/TeX/texbin/$(LATEXMK)),)
    export PATH := /Library/TeX/texbin:$(PATH)
  endif
endif

.PHONY: all pdf clean distclean

all: pdf

pdf:
	$(LATEXMK) -cd $(LATEXMK_COMMON) $(LATEXMKFLAGS) "$(PAPER_DIR)$(MAIN).tex"

clean:
	$(LATEXMK) -cd $(LATEXMK_COMMON) -c "$(PAPER_DIR)$(MAIN).tex"
	rm -f "$(PAPER_DIR)$(MAIN).spl" "$(PAPER_DIR)$(MAIN).synctex.gz"

distclean: clean
	rm -f "$(PAPER_DIR)$(MAIN).pdf"
