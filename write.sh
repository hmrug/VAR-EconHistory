#!/bin/sh

# Remove all output, create output directory, create output subdirectories
rm -rf ./output/*
if [ ! -d ./output ]; then
  mkdir -p ./output;
fi
mkdir -p ./output/figures ./output/data ./output/tables 

# Run python scripts that compute, plot figures, and export figures/tables
./descriptive_statistics.py
./analysis.py

# Compile latex
cd ./text
pdflatex report.tex
mv report.pdf ../
