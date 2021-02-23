#!/usr/bin/env python
# encoding: utf-8

import pandas as pd

out_data = './output/data/'
out_tables = './output/tables/'
out_figures = './output/figures/'

def figure(fig_name,file_name):
    return fig_name.savefig(fname=out_figures+file_name,
                           dpi=600)
def table(tab_name,file_name):
    return print(pd.DataFrame(tab_name).to_latex(),
                 file=open(out_tables+file_name,'a'))
def data(df_name,file_name):
    return pd.DataFrame(df_name).to_csv(out_data+file_name,
                                 index=True,header=True,)
def matrix(mat_name,file_name):
    return pd.DataFrame(mat_name).to_csv(out_data+file_name,
                                 index=True,header=True,)
