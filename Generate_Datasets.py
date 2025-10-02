# Run this script to both datasets
import subprocess

scripts = ["Generate_Easy_Task_Dataset.py",
    "Generate_Harder_Task_Dataset.py"]

for script in scripts:
    print(f"\n>>> Running {script}...")
    subprocess.run(["python", script], check=True)
    print(f"<<< Finished {script}\n")