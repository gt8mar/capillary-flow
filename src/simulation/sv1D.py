"""
Filename: sv1D.py
-----------------
This program makes a couple of simulations using simvascular. 
by: Marcus Forst
"""
import subprocess

command_path = "C:\\Program Files\\SimVascular\\svOneDSolver\\2022-10-04\\svOneDSolver.exe"
args = "C:\\Users\\ejerison\\Marcus\\test2\\ROMSimulations\\oned\\solver_1D.in"

subprocess.call(f"{command_path} {args[0]}" , shell = True, stdout= subprocess.DEVNULL)