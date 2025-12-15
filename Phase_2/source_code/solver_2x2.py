import cv2
import numpy as np
import os
from itertools import permutations

# =====================================================
# 1. CONFIGURATION
# =====================================================
PUZZLE_N = 2
PUZZLE_FOLDER = "D:/Image_Proccessing/Project/Gravity Falls/puzzle_2x2"
CORRECT_FOLDER = "D:/Image_Proccessing/Project/Gravity Falls/correct"
OUTPUT_FOLDER = "results/2x2"

# =====================================================
# 2. SOLVER CLASS
# =====================================================
class JigsawSolver2x2:
    def __init__(self, puzzle_path, correct_path=None):
        self.img_bgr = cv2.imread(puzzle_path)
        if self.img_bgr is None:
            raise ValueError(f"Could not load puzzle: {puzzle_path}")
        
        # Load Correct Image (Ground Truth) if available
        self.correct_bgr = None
        if correct_path and os.path.exists(correct_path):
            self.correct_bgr = cv2.imread(correct_path)
        
        # Split scrambled image
        self.pieces_bgr_scrambled = self._split(self.img_bgr)
        self.pieces_lab_scrambled = [cv2.cvtColor(p, cv2.COLOR_BGR2LAB).astype(np.float32) for p in self.pieces_bgr_scrambled]
        
        # Mapping: Scrambled Index -> True ID (0..3)
        self.id_map = self._map_pieces_to_truth()

    def _split(self, img):
        h, w = img.shape[:2]
        ph, pw = h // PUZZLE_N, w // PUZZLE_N
        pieces = []
        for r in range(PUZZLE_N):
            for c in range(PUZZLE_N):
                pieces.append(img[r*ph:(r+1)*ph, c*pw:(c+1)*pw])
        return pieces

    def _map_pieces_to_truth(self):
        """
        Matches scrambled pieces to the ground truth image to find their Real IDs.
        If no ground truth, assumes 0,1,2,3 (which causes low accuracy).
        """
        if self.correct_bgr is None:
            return {i: i for i in range(4)} # Fallback

        # Split ground truth into perfect pieces 0,1,2,3
        true_pieces = self._split(self.correct_bgr)
        
        # Map each scrambled piece to the closest true piece
        mapping = {}
        used_true_ids = set()
        
        for s_idx, s_piece in enumerate(self.pieces_bgr_scrambled):
            best_diff = float('inf')
            best_true_id = -1
            
            for t_idx, t_piece in enumerate(true_pieces):
                # Simple pixel diff
                if s_piece.shape != t_piece.shape:
                    t_piece = cv2.resize(t_piece, (s_piece.shape[1], s_piece.shape[0]))
                
                diff = np.sum(np.abs(s_piece.astype(int) - t_piece.astype(int)))
                if diff < best_diff:
                    best_diff = diff
                    best_true_id = t_idx
            
            mapping[s_idx] = best_true_id
            
        return mapping

    def _border_cost(self, p1, side1, p2, side2):
        if side1 == 'right':
            edge1, edge2 = p1[:, -1], p2[:, 0]
        else:
            edge1, edge2 = p1[-1, :], p2[0, :]
        return np.sum(np.abs(edge1 - edge2))

    def solve(self):
        best_perm = None
        best_score = float('inf')
        
        # Permutation contains indices of scrambled pieces [0, 1, 2, 3]
        for perm in permutations(range(4)):
            grid = np.array(perm).reshape(2, 2)
            score = 0
            
            # Use LAB pieces for solving
            p = [self.pieces_lab_scrambled[i] for i in perm]
            
            score += self._border_cost(p[0], 'right', p[1], 'left')
            score += self._border_cost(p[2], 'right', p[3], 'left')
            score += self._border_cost(p[0], 'bottom', p[2], 'top')
            score += self._border_cost(p[1], 'bottom', p[3], 'top')
            
            if score < best_score:
                best_score = score
                best_perm = perm
        
        # 1. Reconstruct Image (Visual)
        visual_grid = np.array(best_perm).reshape(2, 2)
        rows = []
        for r in range(PUZZLE_N):
            rows.append(np.hstack([self.pieces_bgr_scrambled[visual_grid[r,c]] for c in range(PUZZLE_N)]))
        solved_img = np.vstack(rows)
        
        # 2. Reconstruct ID Grid (Logical) using the Mapping
        # We convert "Scrambled ID" -> "True ID"
        true_id_grid = np.zeros((2,2), dtype=int)
        for r in range(2):
            for c in range(2):
                scrambled_id = visual_grid[r,c]
                true_id_grid[r,c] = self.id_map[scrambled_id]
                
        return solved_img, true_id_grid

# =====================================================
# 3. ACCURACY CHECKER
# =====================================================
def compute_neighbor_accuracy(grid: np.ndarray) -> float:
    rows, cols = grid.shape
    correct_neighbors = 0
    total_boundaries = 0
    
    for r in range(rows):
        for c in range(cols):
            piece_a = grid[r, c]
            if piece_a == -1: continue
            
            # Right Neighbor
            if c < cols - 1:
                piece_b = grid[r, c + 1]
                total_boundaries += 1
                if piece_b != -1 and piece_a % cols < cols - 1 and piece_b == piece_a + 1:
                    correct_neighbors += 1
            
            # Bottom Neighbor
            if r < rows - 1:
                piece_b = grid[r + 1, c]
                total_boundaries += 1
                if piece_b != -1 and piece_a // cols < rows - 1 and piece_b == piece_a + cols:
                    correct_neighbors += 1
    
    return correct_neighbors / total_boundaries if total_boundaries > 0 else 0.0

# =====================================================
# 4. MAIN EXECUTION
# =====================================================
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
    
    correct_map = {os.path.splitext(f)[0]: f for f in os.listdir(CORRECT_FOLDER)} if os.path.exists(CORRECT_FOLDER) else {}
    puzzle_files = [f for f in os.listdir(PUZZLE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"--- SOLVING {len(puzzle_files)} 2x2 PUZZLES ---")
    print(f"{'Image':<30} | {'Neighbor Acc':<12}")
    print("-" * 45)

    total_acc = 0
    count = 0

    for p_file in puzzle_files:
        try:
            # Pass BOTH puzzle and correct path to the solver
            c_file = correct_map.get(os.path.splitext(p_file)[0])
            c_path = os.path.join(CORRECT_FOLDER, c_file) if c_file else None
            
            solver = JigsawSolver2x2(os.path.join(PUZZLE_FOLDER, p_file), c_path)
            solved_img, true_id_grid = solver.solve()
            
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(p_file)[0]}_solved.png"), solved_img)
            
            # Now true_id_grid contains correct IDs (0,1,2,3) mapped from ground truth
            acc = compute_neighbor_accuracy(true_id_grid) * 100
            total_acc += acc
            count += 1
            
            print(f"{p_file:<30} | {acc:>10.1f}%")
            
        except Exception as e:
            print(f"{p_file:<30} | ERROR: {e}")

    if count > 0:
        print("-" * 45)
        print(f"Average Accuracy: {total_acc/count:.2f}%")