#!/bin/sh

rm -rf ./output/*
if [ ! -d ./output ]; then
  mkdir -p ./output;
fi
mkdir -p ./output/figures ./output/data ./output/tables 

./descriptive_statistics.py
./analysis.py

cd ./text
pdflatex report.tex
mv report.pdf ../
