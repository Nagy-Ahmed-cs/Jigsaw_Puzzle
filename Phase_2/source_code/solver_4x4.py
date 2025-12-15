import cv2
import numpy as np
import os

# =====================================================
# 1. CONFIGURATION
# =====================================================
PUZZLE_N = 4

PUZZLE_FOLDER = "D:/Image_Proccessing/Project/Gravity Falls/puzzle_4x4"      # Folder containing scrambled puzzles
CORRECT_FOLDER = "D:/Image_Proccessing/Project/Gravity Falls/correct"        # Folder containing correct original images
OUTPUT_FOLDER = "results/4x4" 
BEAM_WIDTH = 1010

# =====================================================
# 2. SOLVER CLASS
# =====================================================
class JigsawSolver4x4:
    def __init__(self, image_path):
        self.img_bgr = cv2.imread(image_path)
        if self.img_bgr is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        self.img_lab = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        self.pieces_lab = self._split(self.img_lab)
        self.pieces_bgr = self._split(self.img_bgr)
        self.n = len(self.pieces_lab)
        
        # Precompute costs
        self.H, self.V = self._compute_costs()

    def _split(self, img):
        h, w = img.shape[:2]
        ph, pw = h // PUZZLE_N, w // PUZZLE_N
        return [img[r*ph:(r+1)*ph, c*pw:(c+1)*pw] for r in range(PUZZLE_N) for c in range(PUZZLE_N)]

    def _border_cost(self, p1, side, p2):
        """L1 dissimilarity with gradient term"""
        if side == 'H': # Horizontal (p1 left, p2 right)
            edge1, inner1 = p1[:, -1], p1[:, -2]
            edge2 = p2[:, 0]
        else: # Vertical (p1 top, p2 bottom)
            edge1, inner1 = p1[-1, :], p1[-2, :]
            edge2 = p2[0, :]
        
        color_diff = np.sum(np.abs(edge1 - edge2))
        grad1 = edge1 - inner1
        grad2 = edge2 - edge1
        grad_term = np.sum(np.abs(grad2 - grad1))
        
        return color_diff + 1.5 * grad_term

    def _compute_costs(self):
        H = np.full((self.n, self.n), np.inf)
        V = np.full((self.n, self.n), np.inf)
        
        for i in range(self.n):
            for j in range(self.n):
                if i == j: continue
                H[i,j] = self._border_cost(self.pieces_lab[i], 'H', self.pieces_lab[j])
                V[i,j] = self._border_cost(self.pieces_lab[i], 'V', self.pieces_lab[j])
        
        return H, V

    def solve(self):
        """Beam search assembly"""
        # Start with each piece as potential top-left
        beam = [(0, [i]) for i in range(self.n)]
        
        # Build row by row
        for step in range(self.n - 1):
            candidates = []
            
            for current_cost, current_grid in beam:
                used = set(current_grid)
                next_idx = len(current_grid)
                row = next_idx // PUZZLE_N
                col = next_idx % PUZZLE_N
                
                # Neighbors
                left_neighbor = current_grid[next_idx - 1] if col > 0 else None
                top_neighbor = current_grid[next_idx - PUZZLE_N] if row > 0 else None
                
                for p_cand in range(self.n):
                    if p_cand in used: continue
                    
                    added_cost = 0
                    if left_neighbor is not None:
                        added_cost += self.H[left_neighbor, p_cand]
                    if top_neighbor is not None:
                        added_cost += self.V[top_neighbor, p_cand]
                    
                    candidates.append((current_cost + added_cost, current_grid + [p_cand]))
            
            candidates.sort(key=lambda x: x[0])
            beam = candidates[:BEAM_WIDTH]
        
        # Best solution
        best_grid = np.array(beam[0][1]).reshape(PUZZLE_N, PUZZLE_N)
        
        # Reconstruct
        rows = []
        for r in range(PUZZLE_N):
            rows.append(np.hstack([self.pieces_bgr[best_grid[r,c]] for c in range(PUZZLE_N)]))
        return np.vstack(rows)

# =====================================================
# 3. ACCURACY CHECKER
# =====================================================
def calculate_accuracy(solved_img, correct_img):
    if solved_img.shape != correct_img.shape:
        solved_img = cv2.resize(solved_img, (correct_img.shape[1], correct_img.shape[0]))
    h, w = solved_img.shape[:2]
    ph, pw = h // PUZZLE_N, w // PUZZLE_N
    
    correct_count = 0
    for r in range(PUZZLE_N):
        for c in range(PUZZLE_N):
            y, x = r*ph, c*pw
            diff = np.mean(np.abs(solved_img[y:y+ph, x:x+pw].astype(int) - correct_img[y:y+ph, x:x+pw].astype(int)))
            if diff < 15:
                correct_count += 1
    
    return (correct_count / (PUZZLE_N * PUZZLE_N)) * 100.0

# =====================================================
# 4. MAIN EXECUTION
# =====================================================
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    correct_map = {}
    if os.path.exists(CORRECT_FOLDER):
        for f in os.listdir(CORRECT_FOLDER):
            correct_map[os.path.splitext(f)[0]] = f

    puzzle_files = [f for f in os.listdir(PUZZLE_FOLDER) if f.lower().endswith(('.png', '.jpg'))]
    print(f"--- SOLVING {len(puzzle_files)} 4x4 PUZZLES ---")
    print(f"{'Image':<30} | {'Accuracy':<10}")
    print("-" * 45)

    total_acc, count = 0, 0

    for p_file in puzzle_files:
        try:
            solver = JigsawSolver4x4(os.path.join(PUZZLE_FOLDER, p_file))
            solved = solver.solve()
            
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(p_file)[0]}_solved.png"), solved)
            
            acc = 0
            c_file = correct_map.get(os.path.splitext(p_file)[0])
            if c_file:
                c_img = cv2.imread(os.path.join(CORRECT_FOLDER, c_file))
                if c_img is not None:
                    acc = calculate_accuracy(solved, c_img)
                    total_acc += acc
                    count += 1
            
            print(f"{p_file:<30} | {acc:>5.1f}%")
        except Exception as e:
            print(f"{p_file:<30} | ERROR: {e}")

    if count > 0:
        print("-" * 45)
        print(f"Average Accuracy: {total_acc/count:.2f}%")