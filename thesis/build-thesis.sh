#!/bin/sh
cd ../thesis
shopt -s extglob
git clean -fx -e !(thesis.pdf)
shopt -u extglob
echo Delete build files.

# Copyright (C) 2014-2020 by Thomas Auzinger <thomas@auzinger.name>

# Replace the 'x' in the next line with the name of the thesis' main LaTeX document without the '.tex' extension
SOURCE=thesis

# Build the thesis document
pdflatex $SOURCE
bibtex $SOURCE
bibtex Online
pdflatex $SOURCE
pdflatex $SOURCE
makeindex -t $SOURCE.glg -s $SOURCE.ist -o $SOURCE.gls $SOURCE.glo
makeindex -t $SOURCE.alg -s $SOURCE.ist -o $SOURCE.acr $SOURCE.acn
makeindex -t $SOURCE.ilg -o $SOURCE.ind $SOURCE.idx
pdflatex $SOURCE
pdflatex $SOURCE

echo
echo
echo Thesis document compiled.
cd ../thesis
shopt -s extglob
git clean -fx -e !(thesis.pdf)
shopt -u extglob
echo Delete build files.
