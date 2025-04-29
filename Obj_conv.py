import numpy as np
import subprocess
import os
import argparse

class Mesh:
    def __init__(self, vertices, faces):
        self.vertices = vertices  # shape (N, 3)
        self.faces = faces        # shape (M, 3)

    @staticmethod
    def from_obj(filename):
        vertices = []
        faces = []
        with open(filename, 'r') as file:
            for line in file:
                if line.startswith('v '):
                    parts = line.split()
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith('f '):
                    parts = line.split()
                    faces.append([int(p.split('/')[0]) for p in parts[1:]])
        return Mesh(np.array(vertices, dtype=np.float64), np.array(faces, dtype=int))

    def to_obj(self, filename):
        with open(filename, 'w') as file:
            for v in self.vertices:
                file.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for f in self.faces:
                file.write(f"f {f[0]} {f[1]} {f[2]}\n")

    def export_for_solver(self, folder='data'):
        os.makedirs(folder, exist_ok=True)
        np.savetxt(os.path.join(folder, 'vertices.dat'), self.vertices)
        np.savetxt(os.path.join(folder, 'faces.dat'), self.faces, fmt='%d')

    @staticmethod
    def downsample(mesh, target_vertices):
        if mesh.vertices.shape[0] <= target_vertices:
            return mesh
        idx = np.linspace(0, mesh.vertices.shape[0] - 1, target_vertices).astype(int)
        idx_set = set(idx)
        vertices = mesh.vertices[idx]
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(idx)}
        faces = []
        for f in mesh.faces:
            if all((v - 1) in idx_set for v in f):
                faces.append([old_to_new[v - 1] + 1 for v in f])
        return Mesh(vertices, np.array(faces, dtype=int))

def run_solver():
    subprocess.run(["solver.exe"], check=True)

def read_forces():
    with open('data/forces.dat', 'r') as file:
        line = file.readline().strip()
        # Remove '*' characters if they exist
        line = line.replace('*', '')
        drag, lift = map(float, line.split())
    return drag, lift

def compute_vertex_normals(mesh):
    normals = np.zeros_like(mesh.vertices)
    for face in mesh.faces:
        idx = face - 1
        v1, v2, v3 = mesh.vertices[idx[0]], mesh.vertices[idx[1]], mesh.vertices[idx[2]]
        normal = np.cross(v2 - v1, v3 - v1)
        normal /= np.linalg.norm(normal) + 1e-8
        normals[idx] += normal
    norms = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8
    return normals / norms

def smooth_vertices(mesh, strength=0.5):
    normals = compute_vertex_normals(mesh)
    displacement = normals * strength
    return mesh.vertices + displacement

def optimize_mesh(mesh, objective='minimize_drag', steps=1000, step_size=0.01):
    print(f"Optimization started: {objective}")
    best_vertices = mesh.vertices.copy()
    best_faces = mesh.faces.copy()
    best_mesh = Mesh(best_vertices, best_faces)
    best_drag, best_lift = evaluate_mesh(best_mesh)
    print(f"Initial Drag: {best_drag:.6f}, Lift: {best_lift:.6f}")

    for step in range(steps):
        new_vertices = smooth_vertices(Mesh(best_vertices, best_faces), strength=step_size)
        trial_mesh = Mesh(new_vertices, best_faces)

        trial_drag, trial_lift = evaluate_mesh(trial_mesh)

        improved = False
        if objective == 'minimize_drag' and trial_drag < best_drag:
            improved = True
        elif objective == 'maximize_lift_drag' and (trial_lift / trial_drag) > (best_lift / best_drag):
            improved = True

        if improved:
            best_vertices = new_vertices
            best_drag, best_lift = trial_drag, trial_lift
            print(f"Step {step+1}: Improved -> Drag: {best_drag:.6f}, Lift: {best_lift:.6f}")
        else:
            print(f"Step {step+1}: No improvement.")

    return Mesh(best_vertices, best_faces)

def evaluate_mesh(mesh):
    mesh.export_for_solver('data')
    run_solver()
    return read_forces()

def main():
    parser = argparse.ArgumentParser(description="Aerodynamic Shape Optimizer")
    parser.add_argument('input_obj', type=str, help="Input .obj file")
    parser.add_argument('output_obj', type=str, help="Output .obj file")
    parser.add_argument('--objective', type=str, choices=['minimize_drag', 'maximize_lift_drag'], default='minimize_drag')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--step_size', type=float, default=0.01)
    parser.add_argument('--target_vertices', type=int, default=10000000)
    args = parser.parse_args()

    original_mesh = Mesh.from_obj(args.input_obj)
    mesh = Mesh.downsample(original_mesh, target_vertices=args.target_vertices)

    optimized_mesh = optimize_mesh(mesh, objective=args.objective, steps=args.steps, step_size=args.step_size)

    optimized_mesh.to_obj(args.output_obj)
    print(f"Optimization finished. Saved to {args.output_obj}")

if __name__ == "__main__":
    main()
