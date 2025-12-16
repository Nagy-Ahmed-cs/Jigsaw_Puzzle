import cv2
import numpy as np
import os

# =====================================================
# 1. CONFIGURATION
# =====================================================
PUZZLE_N = 8
PUZZLE_FOLDER = "D:/Image_Proccessing/Project/Gravity Falls/puzzle_8x8"
CORRECT_FOLDER = "D:/Image_Proccessing/Project/Gravity Falls/correct"
OUTPUT_FOLDER = "results/8x8"
BEAM_WIDTH = 500

# =====================================================
# 2. SOLVER CLASS
# =====================================================
class JigsawSolver8x8:
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
        self.n = len(self.pieces_lab_scrambled)
        
        # Mapping: Scrambled Index -> True ID
        self.id_map = self._map_pieces_to_truth()
        
        # Precompute costs
        self.H, self.V = self._compute_costs()

    def _split(self, img):
        h, w = img.shape[:2]
        ph, pw = h // PUZZLE_N, w // PUZZLE_N
        pieces = []
        for r in range(PUZZLE_N):
            for c in range(PUZZLE_N):
                pieces.append(img[r*ph:(r+1)*ph, c*pw:(c+1)*pw])
        return pieces

    def _map_pieces_to_truth(self):
        """Matches scrambled pieces to ground truth IDs"""
        if self.correct_bgr is None:
            return {i: i for i in range(self.n)}

        true_pieces = self._split(self.correct_bgr)
        mapping = {}
        
        for s_idx, s_piece in enumerate(self.pieces_bgr_scrambled):
            best_diff = float('inf')
            best_true_id = -1
            
            for t_idx, t_piece in enumerate(true_pieces):
                if s_piece.shape != t_piece.shape:
                    t_piece = cv2.resize(t_piece, (s_piece.shape[1], s_piece.shape[0]))
                
                diff = np.sum(np.abs(s_piece.astype(int) - t_piece.astype(int)))
                if diff < best_diff:
                    best_diff = diff
                    best_true_id = t_idx
            
            mapping[s_idx] = best_true_id
        return mapping

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
                H[i,j] = self._border_cost(self.pieces_lab_scrambled[i], 'H', self.pieces_lab_scrambled[j])
                V[i,j] = self._border_cost(self.pieces_lab_scrambled[i], 'V', self.pieces_lab_scrambled[j])
        
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
        
        # Best solution (indices of scrambled pieces)
        visual_grid = np.array(beam[0][1]).reshape(PUZZLE_N, PUZZLE_N)
        
        # Reconstruct Visual Image
        rows = []
        for r in range(PUZZLE_N):
            rows.append(np.hstack([self.pieces_bgr_scrambled[visual_grid[r,c]] for c in range(PUZZLE_N)]))
        solved_img = np.vstack(rows)
        
        # Reconstruct ID Grid (Logical) using the Mapping
        true_id_grid = np.zeros((PUZZLE_N, PUZZLE_N), dtype=int)
        for r in range(PUZZLE_N):
            for c in range(PUZZLE_N):
                scrambled_id = visual_grid[r,c]
                true_id_grid[r,c] = self.id_map[scrambled_id]
                
        return solved_img, true_id_grid

# =====================================================
# 3. NEIGHBOR ACCURACY 
# =====================================================
def compute_neighbor_accuracy(grid: np.ndarray) -> float:
    """
    Compute Pairwise Neighbor Accuracy for a reconstructed grid.
    """
    rows, cols = grid.shape
    correct_neighbors = 0
    total_boundaries = 0
    
    for r in range(rows):
        for c in range(cols):
            piece_a = grid[r, c]
            if piece_a == -1: continue
            
            # Check RIGHT neighbor
            if c < cols - 1:
                piece_b = grid[r, c + 1]
                total_boundaries += 1
                if piece_b != -1:
                    if piece_a % cols < cols - 1 and piece_b == piece_a + 1:
                        correct_neighbors += 1
            
            # Check BOTTOM neighbor
            if r < rows - 1:
                piece_b = grid[r + 1, c]
                total_boundaries += 1
                if piece_b != -1:
                    if piece_a // cols < rows - 1 and piece_b == piece_a + cols:
                        correct_neighbors += 1
    
    if total_boundaries == 0:
        return 0.0
    
    return correct_neighbors / total_boundaries

# =====================================================
# 4. MAIN EXECUTION
# =====================================================
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
    
    correct_map = {os.path.splitext(f)[0]: f for f in os.listdir(CORRECT_FOLDER)} if os.path.exists(CORRECT_FOLDER) else {}
    puzzle_files = [f for f in os.listdir(PUZZLE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"--- SOLVING {len(puzzle_files)} 8x8 PUZZLES ---")
    print(f"{'Image':<30} | {'Neighbor Accuracy':<20}")
    print("-" * 55)

    total_neighbor_acc = 0
    count = 0

    for p_file in puzzle_files:
        try:
            c_file = correct_map.get(os.path.splitext(p_file)[0])
            c_path = os.path.join(CORRECT_FOLDER, c_file) if c_file else None
            
            solver = JigsawSolver8x8(os.path.join(PUZZLE_FOLDER, p_file), c_path)
            solved_img, true_id_grid = solver.solve()
            
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(p_file)[0]}_solved.png"), solved_img)
            
            # Accuracy
            neighbor_acc = compute_neighbor_accuracy(true_id_grid) * 100
            total_neighbor_acc += neighbor_acc
            
            count += 1
            print(f"{p_file:<30} | {neighbor_acc:>10.1f}%")
            
        except Exception as e:
            print(f"{p_file:<30} | ERROR: {e}")

    if count > 0:
        print("-" * 55)
        print(f"Average Neighbor Accuracy: {total_neighbor_acc/count:.2f}%")