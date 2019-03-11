import numpy as np
import pandas as pd

df = pd.read_csv('data.csv')

colnames = df.columns

records = len(df)

print("\\begin{table}[h!]")
print("  \\begin{center}")
print("    \\caption{Data Summary}")
print("    \\label{tab:table1}")
print("    \\begin{tabular}{l|l|r|r|r|r} ")
print("      \\textbf{Col Name} & \\textbf{Type} & \\textbf{Missing \%} & \\textbf{Min} & \\textbf{Mean} & \\textbf{Max}\\\\")
print("      \\hline")

for name in colnames:
    nacount = len(df[df[name].isna()])
    napercent = round(100*nacount/records,3)
    valtype = "Char"
    thetype = str(type(df.loc[1,name]))
    if thetype == "<class 'numpy.float64'>" :
       valtype = "Real"
    if thetype == "<class 'numpy.int64'>" :
       valtype = "Int"
    if (valtype != "Char") :
        themin = round(df[name].min(),3)
        themean = round(df[name].mean(),3)
        themax = round(df[name].max(),3)
    else:
        themin = "-"
        themean = "-"
        themax = "-"
    print("      ", name, "&", valtype, "&", napercent, "&", themin, "&", themean, "&", themax, "\\\\")

print("    \\end{tabular}")
print("  \\end{center}")
print("\\end{table}")



