""" This scripts generates table 4 of the paper """
import subprocess
from os import chdir
from shutil import copy as shcopy

chdir("../MP_SPDZ_online")

cmd = "python3 ../run_benchmark.py ../Table_4 -oe 1 --input-dir ../data/real_doa90/"

print("Running benchmarks, this will take several hours...")

try:
    subprocess.run(cmd.split(), capture_output=True)
except:
    print("Something went wrong.")

print("Plotting results...")

chdir("../Table_4")

cmd = "python3 ../plot_benchmark.py . --acm-runtimes --table 4"

subprocess.run(cmd.split())

print("For better readability, it is recommended to view this table using gedit or another GUI text editor "
      "with text wrapping disabled.")

