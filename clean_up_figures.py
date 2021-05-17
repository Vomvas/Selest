import subprocess
import os

print("Cleaning up benchmark results and figures...")

os.chdir("~/musical-guide")

rm_cmds = [
    "rm -rf Fig_*/*.pdf",
    "rm -rf Fig_5/*/bench_outputs/*",
    "rm -rf Table_*/table_*.txt"
    "rm -rf Table_*/bench_outputs/*"
]

for rm_cmd in rm_cmds:
    subprocess.run(rm_cmd.split(), capture_output=True)

print("Figures and results deleted.")
