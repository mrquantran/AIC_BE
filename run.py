import subprocess
import os

# Path to the python file
python_path = "./.venv/Scripts/python.exe"

env = os.environ.copy()

# Command to run locust using python from the virtual environment
command = ["uvicorn", "main:app", "--reload", "--port", "8000"]

# Run the command with the updated environment
subprocess.run(command, env=env)
