
import cv2
import numpy as np
import os

# =====================================================
# 1. CONFIGURATION
# =====================================================
N = 8
PUZZLE_FOLDER = "D:/Image_Proccessing/Project/Gravity Falls/puzzle_8x8"      # Folder containing scrambled puzzles
CORRECT_FOLDER = "D:/Image_Proccessing/Project/Gravity Falls/correct"        # Folder containing correct original images
OUTPUT_FOLDER = "results/8x8" 
EPS = 1e-6

# =====================================================
# 2. ADVANCED SOLVER (Best Buddies + PBC)
# =====================================================
class JigsawSolverBestBuddies:
    def __init__(self, image_path):
        self.img_bgr = cv2.imread(image_path)
        if self.img_bgr is None:
            raise ValueError(f"Could not load {image_path}")
        
        # Paper suggests LAB color space is best
        self.img_lab = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        self.pieces = self._split_image(self.img_lab, N)
        self.pieces_bgr = self._split_image(self.img_bgr, N)
        self.num_pieces = len(self.pieces)

    def _split_image(self, img, N):
        h, w, _ = img.shape
        ph, pw = h // N, w // N
        pieces = []
        for i in range(N):
            for j in range(N):
                pieces.append(img[i*ph:(i+1)*ph, j*pw:(j+1)*pw])
        return pieces

    def _prediction_compatibility(self, p1, p2, relation):
        """
        Implements the Prediction-Based Compatibility (PBC) from the paper.
        It predicts the next row/col of pixels based on gradients.
        """
        # Extract boundaries
        if relation == "horizontal": # p1 Left, p2 Right
            # Predict p2's left edge from p1's right edge
            # Taylor expansion: f(x+1) = f(x) + f'(x)
            # p1_right_edge = p1[:,-1]
            # p1_grad = p1[:,-1] - p1[:,-2]
            # prediction = p1_right_edge + p1_grad
            
            p1_edge = p1[:, -1, :]
            p1_inner = p1[:, -2, :]
            p2_edge = p2[:, 0, :]
            
            prediction = p1_edge + (p1_edge - p1_inner)
            diff = np.mean(np.abs(prediction - p2_edge))
            return diff
            
        elif relation == "vertical": # p1 Top, p2 Bottom
            p1_edge = p1[-1, :, :]
            p1_inner = p1[-2, :, :]
            p2_edge = p2[0, :, :]
            
            prediction = p1_edge + (p1_edge - p1_inner)
            diff = np.mean(np.abs(prediction - p2_edge))
            return diff
        return float('inf')

    def solve(self):
        n = self.num_pieces
        
        # 1. Calculate Pairwise Compatibilities
        # H_cost[i, j] = Cost of (i) being Left of (j)
        # V_cost[i, j] = Cost of (i) being Top of (j)
        H_cost = np.full((n, n), np.inf)
        V_cost = np.full((n, n), np.inf)
        
        for i in range(n):
            for j in range(n):
                if i == j: continue
                H_cost[i,j] = self._prediction_compatibility(self.pieces[i], self.pieces[j], "horizontal")
                V_cost[i,j] = self._prediction_compatibility(self.pieces[i], self.pieces[j], "vertical")

        # 2. Find "Best Buddies"
        # Two pieces are best buddies if they mutually pick each other as best match
        matches = [] # List of (p1, p2, relation, score)
        
        # Horizontal Best Buddies
        for i in range(n):
            best_j = np.argmin(H_cost[i, :]) # Who does i like best?
            best_i_for_j = np.argmin(H_cost[:, best_j]) # Who does j like best?
            
            if i == best_i_for_j: # Mutual love!
                matches.append((i, best_j, "H", H_cost[i, best_j]))
                
        # Vertical Best Buddies
        for i in range(n):
            best_j = np.argmin(V_cost[i, :])
            best_i_for_j = np.argmin(V_cost[:, best_j])
            
            if i == best_i_for_j:
                matches.append((i, best_j, "V", V_cost[i, best_j]))

        # 3. Assemble Clusters
        # Sort matches by reliability (score)
        matches.sort(key=lambda x: x[3])
        
        # We build relative coordinates: map[piece_id] = (r, c)
        # Start with each piece in its own cluster
        clusters = [{i: (0,0)} for i in range(n)] 
        
        for p1, p2, rel, score in matches:
            # Find which clusters p1 and p2 belong to
            c1_idx = -1
            c2_idx = -1
            for idx, clust in enumerate(clusters):
                if p1 in clust: c1_idx = idx
                if p2 in clust: c2_idx = idx
            
            if c1_idx == c2_idx: continue # Already in same cluster
            
            # Merge Cluster 2 into Cluster 1
            clust1 = clusters[c1_idx]
            clust2 = clusters[c2_idx]
            
            # Calculate offset
            r1, c1 = clust1[p1]
            if rel == "H": # p2 is to Right of p1
                r_target, c_target = r1, c1 + 1
            else:          # p2 is Below p1
                r_target, c_target = r1 + 1, c1
            
            # Shift cluster 2 to align with target
            r2_curr, c2_curr = clust2[p2]
            dr, dc = r_target - r2_curr, c_target - c2_curr
            
            # Add all pieces from c2 to c1
            conflict = False
            for p, (r, c) in clust2.items():
                new_pos = (r + dr, c + dc)
                if new_pos in clust1.values(): 
                    conflict = True; break # Overlap!
                clust1[p] = new_pos
                
            if not conflict:
                clusters.pop(c2_idx) # Remove old cluster
        
        # 4. Final Grid Construction
        # We take the largest cluster
        largest_cluster = max(clusters, key=len)
        
        # Normalize coordinates
        rs = [pos[0] for pos in largest_cluster.values()]
        cs = [pos[1] for pos in largest_cluster.values()]
        min_r, min_c = min(rs), min(cs)
        
        grid = np.full((N, N), -1, dtype=int)
        used_pieces = set()
        
        for p, (r, c) in largest_cluster.items():
            final_r, final_c = r - min_r, c - min_c
            if 0 <= final_r < N and 0 <= final_c < N:
                grid[final_r, final_c] = p
                used_pieces.add(p)
                
        # Fill remaining holes (Greedy fallback)
        remaining = list(set(range(n)) - used_pieces)
        for r in range(N):
            for c in range(N):
                if grid[r,c] == -1 and remaining:
                    grid[r,c] = remaining.pop(0) # Simple fallback
                    
        # Reconstruct Image
        rows_imgs = []
        for r in range(N):
            row_pieces = []
            for c in range(N):
                idx = grid[r,c]
                row_pieces.append(self.pieces_bgr[idx])
            rows_imgs.append(np.hstack(row_pieces))
            
        return np.vstack(rows_imgs)

# =====================================================
# 3. ACCURACY CHECKER (Same as before)
# =====================================================
def calculate_accuracy(solved_img, correct_img, N):
    if solved_img.shape != correct_img.shape:
        solved_img = cv2.resize(solved_img, (correct_img.shape[1], correct_img.shape[0]))
    h, w, _ = solved_img.shape
    ph, pw = h // N, w // N
    correct_count = 0
    for r in range(N):
        for c in range(N):
            y, x = r * ph, c * pw
            diff = np.mean(np.abs(solved_img[y:y+ph, x:x+pw].astype(int) - correct_img[y:y+ph, x:x+pw].astype(int)))
            if diff < 15: correct_count += 1
    return (correct_count / (N*N)) * 100.0

# =====================================================
# 4. MAIN LOOP
# =====================================================
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
    
    correct_map = {}
    if os.path.exists(CORRECT_FOLDER):
        for f in os.listdir(CORRECT_FOLDER):
            correct_map[os.path.splitext(f)[0]] = f
            
    files = [f for f in os.listdir(PUZZLE_FOLDER) if f.lower().endswith(('.png', '.jpg'))]
    print(f"--- BEST BUDDIES SOLVER ({len(files)} puzzles) ---")
    
    total_acc = 0
    count = 0
    
    for f in files:
        try:
            solver = JigsawSolverBestBuddies(os.path.join(PUZZLE_FOLDER, f))
            res = solver.solve()
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, f.replace(".jpg",".png")), res)
            
            c_name = correct_map.get(os.path.splitext(f)[0])
            acc = 0
            if c_name:
                acc = calculate_accuracy(res, cv2.imread(os.path.join(CORRECT_FOLDER, c_name)), N)
                total_acc += acc
                count += 1
            
            print(f"{f:<20} | {acc:>5.1f}%")
        except Exception as e:
            print(f"Error {f}: {e}")
            
    if count: print(f"Avg: {total_acc/count:.2f}%")