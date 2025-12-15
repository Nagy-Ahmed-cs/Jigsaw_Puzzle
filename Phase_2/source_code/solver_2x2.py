import cv2
import numpy as np
import os
from itertools import permutations

# =====================================================
# 1. CONFIGURATION
# =====================================================
PUZZLE_N = 2
PUZZLE_FOLDER = "D:/Image_Proccessing/Project/Gravity Falls/puzzle_2x2"      # Folder containing scrambled puzzles
CORRECT_FOLDER = "D:/Image_Proccessing/Project/Gravity Falls/correct"        # Folder containing correct original images
OUTPUT_FOLDER = "results/2x2" 

# =====================================================
# 2. SOLVER CLASS
# =====================================================
class JigsawSolver2x2:
    def __init__(self, image_path):
        self.img_bgr = cv2.imread(image_path)
        if self.img_bgr is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # LAB color space for better perception
        self.img_lab = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        self.pieces_lab = self._split(self.img_lab)
        self.pieces_bgr = self._split(self.img_bgr)

    def _split(self, img):
        h, w = img.shape[:2]
        ph, pw = h // PUZZLE_N, w // PUZZLE_N
        pieces = []
        for r in range(PUZZLE_N):
            for c in range(PUZZLE_N):
                pieces.append(img[r*ph:(r+1)*ph, c*pw:(c+1)*pw])
        return pieces

    def _border_cost(self, p1, side1, p2, side2):
        """L1 Cost with gradient continuity"""
        if side1 == 'right':
            edge1 = p1[:, -1]
            edge2 = p2[:, 0]
        else: # bottom
            edge1 = p1[-1, :]
            edge2 = p2[0, :]
        
        # Simple L1 distance (robust for 2x2)
        return np.sum(np.abs(edge1 - edge2))

    def solve(self):
        """Brute force all 24 permutations (perfect for 2x2)"""
        best_perm = None
        best_score = float('inf')
        
        for perm in permutations(range(4)):
            grid = np.array(perm).reshape(2, 2)
            score = 0
            
            # Horizontal edges
            score += self._border_cost(self.pieces_lab[grid[0,0]], 'right', self.pieces_lab[grid[0,1]], 'left')
            score += self._border_cost(self.pieces_lab[grid[1,0]], 'right', self.pieces_lab[grid[1,1]], 'left')
            
            # Vertical edges
            score += self._border_cost(self.pieces_lab[grid[0,0]], 'bottom', self.pieces_lab[grid[1,0]], 'top')
            score += self._border_cost(self.pieces_lab[grid[0,1]], 'bottom', self.pieces_lab[grid[1,1]], 'top')
            
            if score < best_score:
                best_score = score
                best_perm = perm
        
        # Reconstruct
        grid = np.array(best_perm).reshape(2, 2)
        rows = []
        for r in range(PUZZLE_N):
            rows.append(np.hstack([self.pieces_bgr[grid[r,c]] for c in range(PUZZLE_N)]))
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
    print(f"--- SOLVING {len(puzzle_files)} 2x2 PUZZLES ---")
    print(f"{'Image':<30} | {'Accuracy':<10}")
    print("-" * 45)

    total_acc, count = 0, 0

    for p_file in puzzle_files:
        try:
            solver = JigsawSolver2x2(os.path.join(PUZZLE_FOLDER, p_file))
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