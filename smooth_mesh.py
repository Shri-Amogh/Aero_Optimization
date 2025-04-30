# smooth_mesh.py

import numpy as np

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

def cotangent_weights(vertices, faces):
    n = len(vertices)
    weights = [{} for _ in range(n)]
    for face in faces:
        i, j, k = face[0] - 1, face[1] - 1, face[2] - 1
        for a, b, c in [(i, j, k), (j, k, i), (k, i, j)]:
            va, vb, vc = vertices[a], vertices[b], vertices[c]
            u = va - vc
            v = vb - vc
            cross = np.cross(u, v)
            norm_cross = np.linalg.norm(cross) + 1e-8
            dot = np.dot(u, v)
            cot_angle = dot / norm_cross
            weights[a][b] = weights[a].get(b, 0) + cot_angle
            weights[b][a] = weights[b].get(a, 0) + cot_angle
    return weights

def cotangent_laplacian_smooth(vertices, weights, strength=0.05):
    new_vertices = vertices.copy()
    for i, neighbors in enumerate(weights):
        if not neighbors:
            continue
        weight_sum = sum(neighbors.values())
        if weight_sum == 0:
            continue
        laplacian = np.zeros(3, dtype=np.float64)
        for j, w in neighbors.items():
            laplacian += w * (vertices[j] - vertices[i])
        laplacian /= weight_sum
        new_vertices[i] += strength * laplacian
    return new_vertices

def smooth_surface(vertices, faces, normal_strength=0.05, laplacian_strength=0.2):
    normals = compute_vertex_normals(vertices, faces)
    displaced = vertices - normal_strength * normals
    weights = cotangent_weights(displaced, faces)
    smoothed = cotangent_laplacian_smooth(displaced, weights, strength=laplacian_strength)
    return smoothed

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Post-process mesh smoothing")
    parser.add_argument("input_obj", type=str, help="Input optimized .obj file")
    parser.add_argument("output_obj", type=str, help="Output smoothed .obj file")
    parser.add_argument("--normal_strength", type=float, default=0.001)
    parser.add_argument("--laplacian_strength", type=float, default=0.001)
    parser.add_argument("--iterations", type=int, default=10, help="Number of smoothing iterations")
    args = parser.parse_args()

    mesh = Mesh.from_obj(args.input_obj)
    vertices = mesh.vertices.copy()
    for _ in range(args.iterations):
        vertices = smooth_surface(vertices, mesh.faces,
                                  normal_strength=args.normal_strength,
                                  laplacian_strength=args.laplacian_strength)

    smoothed_mesh = Mesh(vertices, mesh.faces)
    smoothed_mesh.to_obj(args.output_obj)
    print(f"Smoothed mesh saved to {args.output_obj}")

if __name__ == "__main__":
    main()
