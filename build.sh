#!/bin/sh
python3 descriptive_statistics.py
python3 analysis.py
cd text
pdflatex main.tex
mv main.pdf ../
