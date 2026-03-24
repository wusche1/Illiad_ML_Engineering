TEXBIN = /Library/TeX/texbin
LECTURES = $(sort $(wildcard lectures/*/))

lecture-%: lectures/%/slides.tex
	cd lectures/$* && mkdir -p output && \
	$(TEXBIN)/pdflatex -shell-escape -synctex=1 -interaction=nonstopmode -file-line-error -output-directory=output slides.tex && \
	$(TEXBIN)/biber --output-directory=output slides && \
	$(TEXBIN)/pdflatex -shell-escape -synctex=1 -interaction=nonstopmode -file-line-error -output-directory=output slides.tex && \
	$(TEXBIN)/pdflatex -shell-escape -synctex=1 -interaction=nonstopmode -file-line-error -output-directory=output slides.tex

all: $(patsubst lectures/%/,lecture-%,$(LECTURES))

update-links:
	uv run python scripts/tools/update_colab_links.py

clean:
	rm -rf lectures/*/output

.PHONY: all clean update-links
