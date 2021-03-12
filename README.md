# About

Reproducible research project for the class: Matlab & Python: Applications in Economic History at UZH.

Investigation of economic fluctuations in the interwar USA. Were these supply or demand driven? To answer that question a VAR model on Industrial Production and CPI is applied.

Data: 01/1919 - 12/1939, monthly. 

# Data Source

- Statistisches Handbuch der Weltwirtschaft (1936), bearb. im statistischen Reichsamt Verlag für Sozialpolitik, Wirtschaft und Statistik GmbH, Berlin

## Requirements

- Python
    - Numpy 1.19.2+
    - Pandas 1.2.2+
    - Scipy 1.6.1+
    - Matplotlib 3.3.4+
- Full-fledged latex
    - Debian-based linux: apt-get install texlive-full
    - Arch-based linux: pacman -S texlive-most texlive-lang
    - MacOS: https://tug.org/mactex/
    - Windows: https://miktex.org/download/

Can be easily reproduced on any POSIX compliant system (linux, macOS) by running the shell script *write.sh*.

# How to reproduce

1. Install python, latex and all dependencies.
    - All python dependencies should be available in the base version of anaconda;
    - For miniconda/anaconda all packages can be installed by creating a new environment from *environment.yml* file. Command: *conda env create -f environment.yml*;
2. Make *write.sh* file executable (*chmod +x write.sh* in the terminal) and run it by typing *./write.sh*.

If you run windows:
    1. Remove everything in *output/tables/*, but don't delete the directory;
    2. Run *descriptive_statistics.py* and then *analysis.py*;
    3. Go to *text/* directory and run *report.tex* in your latex IDE or with pdflatex. By doing this the final file *report.pdf* will be generated.

# Why I don't use Waf

I wanted to do the whole thing in python instead of matlab and I didn't know how to configure Waf to read my python code so I just figured I can build my own framework for a reproducible project. The structure is very simple. In *tools* directory I created two scripts that help with cleaning the data and exporting figures and tables. These are not needed as the dataset is very simple and exporting can by done directly with pandas, but may be useful if the data was more complicated. Python scripts read files from the data directory and produce all the relevant material in the *output* folder. Then, in the *text* folder all latex files are located. The downside to this approach is that I don't have a script for a windows operating system, and so for windows it must be done manually. The upside is that the structure of this project is very simple. All figures and tables are dynamically generated from the code, so it' basically the same thing.
