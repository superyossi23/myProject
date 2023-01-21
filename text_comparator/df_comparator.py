"""
2022.01.13  v000
2022.01.21  df.loc -> df.iloc

"""

import os
import pandas as pd
import time

dir_path = r'C:\Users\A\Desktop\pythonProject\myProject\text_comparator'
# dir_path = dir_path.replace('\\', '/')

f1 = 'test1.csv'
f2 = 'test2.csv'
F1 = os.path.join(dir_path, f1)
F2 = os.path.join(dir_path, f2)

df1 = pd.read_csv(F1, skiprows=0)
df2 = pd.read_csv(F2, skiprows=0)

dict1 = df1['letter'].to_dict()
dict2 = df2['letter'].to_dict()

start = time.time()
# Find df1-values found in df2
found1 = []
for row in df1.iterrows():  # row = tuple(index, (col_values))
    for val in dict2.values():
        if row[1][0] == val:  ## SELECT ##
            found1.append(row[0])

# Find df2-values found in df1
found2 = []
for row in df2.iterrows():  # row = tuple(index, (col_values))
    for val in dict1.values():
        if row[1][0] == val:  ## SELECT ##
            found2.append(row[0])


# function definition
# https://www.geeksforgeeks.org/highlight-pandas-dataframes-specific-columns-using-apply/
def highlight_cols(x, found):
    # copy df to new - original data is not changed
    df = x.copy()
    # select all values to green color
    df.iloc[:, :] = 'background-color: None'
    # Red for found cells
    for i in found:
        df.iloc[i, 0] = 'background-color: red'  ## SELECT ##
    # return color df
    return df


def set_styles():
    return [dict(selector="th",
                 props=[("font-size", "12pt"), ("background-color", "FFFFFF")]),
            dict(selector="td",
                 props=[('padding', "2px 6px")]),
            dict(selector="tr:nth-child(odd)",
                 props=[("background-color", "f2f3f4")]),
            # dict(selector="tr:hover",
            #      props=[("background-color", "yellow")]),
            ]


styled1 = df1.style\
    .set_properties(**{'max-width': '800px', 'font-size': '12pt', 'text-align': 'left'})\
    .set_precision(2)\
    .set_table_styles(set_styles())\

styled2 = df2.style\
    .set_properties(**{'max-width': '800px', 'font-size': '12pt', 'text-align': 'left'})\
    .set_precision(2)\
    .set_table_styles(set_styles())\


print("Highlight df1 ...")
styled1 = styled1.apply(highlight_cols, found=found1, axis=None)  # axisはデフォルトは0なので列に対して適用。1だと行、Noneだと全体。

print("Highlight df2 ...")
styled2 = styled2.apply(highlight_cols, found=found2, axis=None)  # axisはデフォルトは0なので列に対して適用。1だと行、Noneだと全体。


# EXPORT
rendered1 = styled1.render()  # Styler.render() で Style オブジェクトを HTML テーブルとして出力できます。
rendered2 = styled2.render()  # Styler.render() で Style オブジェクトを HTML テーブルとして出力できます。
with open('df_comparator.html', 'w') as f:
    f.write(f1 + ' :Highlight cells found in ' + f2)
    f.write(rendered1)
    f.write('<br>')
    f.write(f2 + ' :Highlight cells found in ' + f1)
    f.write(rendered2)
    f.close()


end = time.time()
elapsed = end - start
print('df_comparator.py done!! "df_comparator.html" saved.')
print('Elapsed time:', round(elapsed, 2), 'sec.')

