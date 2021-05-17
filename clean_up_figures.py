""" Clean up figures """
import subprocess
import os

print("Cleaning up benchmark results and figures...")

os.chdir("/home/selest/musical-guide")

rm_cmds = [
    "sudo rm -rf Fig_*/*.pdf",
    "sudo rm -rf Fig_5/*/bench_outputs/*",
    "sudo rm -rf Table_*/table_*.txt"
    "sudo rm -rf Table_*/bench_outputs/*"
]

for rm_cmd in rm_cmds:
    subprocess.run(rm_cmd.split(), capture_output=True)

print("Figures and results deleted.")
