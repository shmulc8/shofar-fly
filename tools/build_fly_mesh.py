"""Decimate the flybody OBJ meshes into one compact binary for the Shofar stage.

Output: stage/fly.bin
  uint32 nVerts, uint32 nIndices
  float32 pos[nVerts*3], float32 nrm[nVerts*3], uint8 rgba[nVerts*4], uint32 idx[nIndices]
Vertex colors encode the body part; alpha < 255 marks translucent wing membranes.
"""
import sys, struct, glob, os
import numpy as np
import trimesh, pyfqmr

SRC = sys.argv[1]
OUT = sys.argv[2]
TARGET_FACES = 150_000

SKIP = {'blender_model'}
def color_for(name):
    if name == 'head_red': return (205, 40, 30, 255)          # compound eyes
    if name.endswith('_black'): return (38, 30, 26, 255)
    if name == 'head_ocelli': return (230, 120, 60, 255)
    if 'membrane' in name: return (200, 215, 235, 70)
    if name.endswith('_brown'): return (120, 85, 50, 255)
    if 'wing' in name: return (150, 130, 100, 200)
    if '_lower' in name: return (160, 120, 70, 255)
    return (196, 160, 105, 255)                                 # chitin

files = sorted(f for f in glob.glob(os.path.join(SRC, '*.obj')) if os.path.splitext(os.path.basename(f))[0] not in SKIP)
meshes = []
total_faces = 0
for f in files:
    m = trimesh.load(f, force='mesh', process=False)
    if not isinstance(m, trimesh.Trimesh) or len(m.faces) == 0: continue
    meshes.append((os.path.splitext(os.path.basename(f))[0], m))
    total_faces += len(m.faces)
print(f'{len(meshes)} meshes, {total_faces:,} faces -> target {TARGET_FACES:,}')

parts = []
for name, m in meshes:
    keep = max(300, int(len(m.faces) * TARGET_FACES / total_faces))
    if len(m.faces) > keep:
        s = pyfqmr.Simplify()
        s.setMesh(m.vertices, m.faces)
        s.simplify_mesh(target_count=keep, aggressiveness=6, preserve_border=True, verbose=0)
        v, fcs, _ = s.getMesh()
        m = trimesh.Trimesh(v, fcs, process=False)
    pass
    parts.append((name, m, color_for(name)))

pos, nrm, col, idx, off = [], [], [], [], 0
for name, m, c in parts:
    pos.append(m.vertices.astype(np.float32)); nrm.append(m.vertex_normals.astype(np.float32))
    col.append(np.tile(np.array(c, np.uint8), (len(m.vertices), 1)))
    idx.append((m.faces + off).astype(np.uint32)); off += len(m.vertices)
pos = np.concatenate(pos); nrm = np.concatenate(nrm); col = np.concatenate(col); idx = np.concatenate(idx).ravel()
ctr = (pos.min(0) + pos.max(0)) / 2; pos -= ctr; pos /= np.abs(pos).max()  # unit-ish, centred
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'wb') as fh:
    fh.write(struct.pack('<II', len(pos), len(idx)))
    fh.write(pos.tobytes()); fh.write(nrm.tobytes()); fh.write(col.tobytes()); fh.write(idx.tobytes())
print(f'wrote {OUT}: {len(pos):,} verts, {len(idx)//3:,} faces, {os.path.getsize(OUT)/1e6:.1f} MB')
print('extent', pos.min(0), pos.max(0))
