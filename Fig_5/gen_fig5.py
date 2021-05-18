""" This scripts generates figure 5 of the paper """
import subprocess
from os import chdir
from shutil import copy as shcopy

chdir("../MP_SPDZ_online")

cmd = "python3 ../run_benchmark.py ../Fig_5/a ../Fig_5/b ../Fig_5/c ../Fig_5/d -o --input-dir ../data/real_doa90/"

print("Running benchmarks, this will take some time...")

try:
    subprocess.run(cmd.split(), capture_output=True)
except:
    print("Something went wrong.")

print("Plotting results...")

cmd = "python3 ../plot_benchmark.py ../Fig_5/a ../Fig_5/b ../Fig_5/c ../Fig_5/d"

subprocess.run(cmd.split())

chdir("../Fig_5")

for bench in ["a", "b", "c", "d"]:
    shcopy(f"{bench}/bench_outputs/plot_{bench}.pdf", f"{bench}.pdf")
    print(f"Figure {bench}.pdf saved successfully.")
