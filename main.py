import subprocess
import os

def main():
    input_file = "input.obj"
    output_file = "output.obj"
    objective = "minimize_drag"

    command = ["python", "optimizer.py", input_file, output_file, "--objective", objective]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Optimization failed with error code {e.returncode}")

if __name__ == "__main__":
    main()
       
    file_path1 = "Data/faces.dat"
    os.remove(file_path1)
    print(f"File '{file_path1}' deleted successfully.")
    file_path2 = "Data/forces.dat"
    os.remove(file_path2)
    print(f"File '{file_path2}' deleted successfully.")
    file_path3 = "Data/vertices.dat"
    os.remove(file_path3)
    print(f"File '{file_path3}' deleted successfully.")



