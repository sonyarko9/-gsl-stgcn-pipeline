import numpy as np

def build_mediapipe_adjacency():
    """
    Builds adjacency matrix A for MediaPipe Holistic skeleton.
    75 nodes: 33 pose + 21 left hand + 21 right hand
    """
    num_nodes = 75
    edges = []

    # --- Pose edges (MediaPipe 33 landmarks) ---
    pose_edges = [
        (0,1),(1,2),(2,3),(3,7),       # left face
        (0,4),(4,5),(5,6),(6,8),       # right face
        (9,10),                         # mouth
        (11,12),                        # shoulders
        (11,13),(13,15),               # left arm
        (12,14),(14,16),               # right arm
        (15,17),(15,19),(17,19),       # left hand base
        (16,18),(16,20),(18,20),       # right hand base
        (11,23),(12,24),(23,24),       # torso
        (23,25),(25,27),(27,29),(29,31),(27,31),  # left leg
        (24,26),(26,28),(28,30),(30,32),(28,32),  # right leg
    ]
    edges.extend(pose_edges)

    # --- Left hand edges (nodes 33–53, MediaPipe 21 landmarks) ---
    lh_offset = 33
    lh_edges = [
        (0,1),(1,2),(2,3),(3,4),       # thumb
        (0,5),(5,6),(6,7),(7,8),       # index
        (0,9),(9,10),(10,11),(11,12),  # middle
        (0,13),(13,14),(14,15),(15,16),# ring
        (0,17),(17,18),(18,19),(19,20),# pinky
        (5,9),(9,13),(13,17),          # palm cross
    ]
    edges.extend([(u+lh_offset, v+lh_offset) for u,v in lh_edges])

    # --- Right hand edges (nodes 54–74) ---
    rh_offset = 54
    edges.extend([(u+rh_offset, v+rh_offset) for u,v in lh_edges])

    # Connect wrists to pose landmarks (15=left wrist, 16=right wrist)
    edges.append((15, 33))   # pose left wrist → left hand root
    edges.append((16, 54))   # pose right wrist → right hand root

    # Build symmetric adjacency matrix
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for u, v in edges:
        A[u][v] = 1
        A[v][u] = 1

    # Add self-loops
    np.fill_diagonal(A, 1)

    # Normalize: D^(-1/2) * A * D^(-1/2)
    D = np.diag(A.sum(axis=1) ** -0.5)
    A = D @ A @ D

    return A