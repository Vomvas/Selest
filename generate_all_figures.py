""" TL;DR script to generate all figures and tables for replicability purposes """
#!/usr/bin/python3
import os
import subprocess

print("Generating Figure 3...")
os.chdir("Fig_3")
cmd = "python3 gen_fig3.py"
subprocess.run(cmd.split())

print("Generating Figure 4...")
os.chdir("Fig_4")
cmd = "python3 gen_fig4.py"
subprocess.run(cmd.split())

print("Generating Figure 5...")
os.chdir("Fig_5")
cmd = "python3 gen_fig5.py"
subprocess.run(cmd.split())

print("Generating Table 1...")
os.chdir("Table_1")
cmd = "python3 gen_tab1.py"
subprocess.run(cmd.split())

print("Generating Table 2...")
os.chdir("Table_2")
cmd = "python3 gen_tab2.py"
subprocess.run(cmd.split())

print("Generating Table 3...")
os.chdir("Table_3")
cmd = "python3 gen_tab3.py"
subprocess.run(cmd.split())

print("Generating Table 4...")
os.chdir("Table_4")
cmd = "python3 gen_tab4.py"
subprocess.run(cmd.split())

print("All figures and tables should be generated in their respective directories.")
