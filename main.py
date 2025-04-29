import subprocess
import os

def main():
    input_file = "input.obj"
    output_file = "output.obj"
    final_file = "final.obj"
    objective = "minimize_drag"

    command = ["python", "optimizer.py", input_file, output_file, "--objective", objective]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Optimization failed with error code {e.returncode}")

    command = ["python", "smooth_mesh.py" ,output_file,final_file]


    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Smoothing failed with error code {e.returncode}")

def remove(filename):
    os.remove(filename)
    print (f"File '{filename}' deleted successfully.")




if __name__ == "__main__":
    main()
       
    remove("Data/faces.dat")
    remove("Data/forces.dat")
    remove("Data/vertices.dat")
    remove("output.obj")
    



