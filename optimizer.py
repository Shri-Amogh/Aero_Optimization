import numpy as np
import subprocess
import os

class Mesh:
    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces

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
    def downsample(mesh, target_vertices=1e12):
        if mesh.vertices.shape[0] <= target_vertices:
            return mesh
        idx = np.linspace(0, mesh.vertices.shape[0] - 1, int(target_vertices)).astype(int)
        idx_set = set(idx)
        vertices = mesh.vertices[idx]
        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(idx)}
        faces = []
        for f in mesh.faces:
            if all((v - 1) in idx_set for v in f):
                faces.append([old_to_new[v - 1] + 1 for v in f])
        return Mesh(vertices, np.array(faces, dtype=int))

def run_solver():
    subprocess.run(["Fortran Math/solver.exe"], check=True)

def read_forces():
    with open('data/forces.dat', 'r') as file:
        line = file.readline().replace('*', '0')
        drag, lift = map(float, line.split())
    return drag, lift

def compute_vertex_normals(vertices, faces):
    normals = np.zeros(vertices.shape, dtype=np.float64)
    for f in faces:
        v1, v2, v3 = vertices[f[0]-1], vertices[f[1]-1], vertices[f[2]-1]
        face_normal = np.cross(v2 - v1, v3 - v1)
        face_normal /= np.linalg.norm(face_normal) + 1e-8
        for idx in f:
            normals[idx-1] += face_normal
    norms = np.linalg.norm(normals, axis=1).reshape(-1, 1) + 1e-8
    normals /= norms
    return normals

def estimate_curvature(vertices, faces):
    curvature = np.zeros(vertices.shape[0])
    neighbor_map = {i: set() for i in range(vertices.shape[0])}
    for f in faces:
        for i in range(3):
            neighbor_map[f[i]-1].update([f[(i+1)%3]-1, f[(i+2)%3]-1])

    for i, neighbors in neighbor_map.items():
        if neighbors:
            diff = vertices[i] - np.mean(vertices[list(neighbors)], axis=0)
            curvature[i] = np.linalg.norm(diff)
    return curvature / (np.max(curvature) + 1e-8)

def laplacian_smooth(vertices, faces, strength=0.05):
    neighbor_map = {i: set() for i in range(vertices.shape[0])}
    for f in faces:
        for i in range(3):
            neighbor_map[f[i]-1].update([f[(i+1)%3]-1, f[(i+2)%3]-1])

    new_vertices = vertices.copy()
    for i, neighbors in neighbor_map.items():
        if neighbors:
            avg = np.mean(vertices[list(neighbors)], axis=0)
            new_vertices[i] = vertices[i] * (1 - strength) + avg * strength

    return new_vertices

def smooth_surface(vertices, faces, normal_strength=0.001, laplacian_strength=0.05):
    normals = compute_vertex_normals(vertices, faces)
    curvature = estimate_curvature(vertices, faces)
    adjusted_normals = normals * curvature[:, np.newaxis]
    displaced = vertices - normal_strength * adjusted_normals
    displaced = laplacian_smooth(displaced, faces, strength=laplacian_strength)
    return displaced

def optimize_mesh(mesh, objective='minimize_drag', steps=5000, step_size=0.001):
    print(f"Starting optimization: {objective}")
    base_vertices = mesh.vertices.copy()
    best_mesh = Mesh(base_vertices, mesh.faces.copy())
    best_drag, best_lift = evaluate_mesh(best_mesh)
    print(f"Initial Drag: {best_drag:.6f}, Lift: {best_lift:.6f}")

    no_improvement_count = 0

    for step in range(steps):
        displaced_vertices = smooth_surface(base_vertices, mesh.faces,
                                            normal_strength=step_size,
                                            laplacian_strength=0.1 * step_size)
        trial_mesh = Mesh(displaced_vertices, mesh.faces.copy())
        trial_drag, trial_lift = evaluate_mesh(trial_mesh)

        improve = False
        if objective == 'minimize_drag' and trial_drag < best_drag:
            improve = True
        elif objective == 'maximize_lift_drag' and (trial_lift / trial_drag) > (best_lift / best_drag):
            improve = True

        if improve:
            best_mesh = Mesh(displaced_vertices.copy(), mesh.faces.copy())
            best_drag, best_lift = trial_drag, trial_lift
            base_vertices = displaced_vertices.copy()
            no_improvement_count = 0
            print(f"Step {step+1}: Improved! Drag: {best_drag:.6f}, Lift: {best_lift:.6f}")
        else:
            no_improvement_count += 1
            print(f"Step {step+1}: No improvement.")
            if no_improvement_count >= 20:
                print("Stopping: No improvement in last 20 steps.")
                break

    return best_mesh

def evaluate_mesh(mesh):
    mesh.export_for_solver('data')
    run_solver()
    return read_forces()

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Aerodynamic Shape Optimizer")
    parser.add_argument('input_obj', type=str, help="Input .obj file")
    parser.add_argument('output_obj', type=str, help="Output .obj file")
    parser.add_argument('--objective', type=str, choices=['minimize_drag', 'maximize_lift_drag'], default='minimize_drag')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--step_size', type=float, default=0.01)
    parser.add_argument('--target_vertices', type=int, default=int(1e20))
    args = parser.parse_args()

    original_mesh = Mesh.from_obj(args.input_obj)
    mesh = Mesh.downsample(original_mesh, target_vertices=args.target_vertices)

    optimized_mesh = optimize_mesh(mesh, objective=args.objective, steps=args.steps, step_size=args.step_size)

    optimized_mesh.to_obj(args.output_obj)
    print(f"Optimization finished! Output saved to {args.output_obj}")

if __name__ == "__main__":
    main()
