table2_str = "Note: Tables 2 and 3 of the initial submission were inadvertently identical. This will be fixed\n" \
             "for the camera ready submission.\n\n" \
             "The results in table 2 (and 3) depend directly on the results of table 4.\n" \
             "In order to avoid redundant " \
             "experiments, table 2 (and 3) is incorporated in table 4 for the purposes of the replicability label.\n" \
             "Please find it concatenated in table 4 (last 4 columns)."

print(table2_str)

with open("table_3.txt", "w") as f:
    f.write(table2_str)

