import cv2
import numpy as np
import os
import heapq

# =====================================================
# 1. CONFIGURATION
# =====================================================
PUZZLE_N = 8
PUZZLE_FOLDER = "D:/Image_Proccessing/Project/Gravity Falls/puzzle_8x8"      # Folder containing scrambled puzzles
CORRECT_FOLDER = "D:/Image_Proccessing/Project/Gravity Falls/correct"        # Folder containing correct original images
OUTPUT_FOLDER = "results/8x8" 

# =====================================================
# 2. MEMORY-EFFICIENT A* SOLVER
# =====================================================
class EfficientAStarSolver8x8:
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
        self.best_buddy_costs = self._compute_best_buddies()

    def _split(self, img):
        h, w = img.shape[:2]
        ph, pw = h // PUZZLE_N, w // PUZZLE_N
        return [img[r*ph:(r+1)*ph, c*pw:(c+1)*pw] for r in range(PUZZLE_N) for c in range(PUZZLE_N)]

    def _border_cost(self, p1, side, p2):
        if side == 'H':
            edge1, inner1 = p1[:, -1], p1[:, -2]
            edge2 = p2[:, 0]
        else:
            edge1, inner1 = p1[-1, :], p1[-2, :]
            edge2 = p2[0, :]
        
        color_diff = np.sum(np.abs(edge1 - edge2))
        grad1 = edge1 - inner1
        grad2 = edge2 - edge1
        grad_diff = np.sum(np.abs(grad2 - grad1))
        
        return color_diff + 1.5 * grad_diff

    def _compute_costs(self):
        H = np.full((self.n, self.n), np.inf)
        V = np.full((self.n, self.n), np.inf)
        
        for i in range(self.n):
            for j in range(self.n):
                if i == j: continue
                H[i,j] = self._border_cost(self.pieces_lab[i], 'H', self.pieces_lab[j])
                V[i,j] = self._border_cost(self.pieces_lab[i], 'V', self.pieces_lab[j])
        
        return H, V

    def _compute_best_buddies(self):
        """Precompute best possible match for each piece"""
        best = np.zeros(self.n)
        for i in range(self.n):
            min_cost = np.inf
            for j in range(self.n):
                if i != j:
                    min_cost = min(min_cost, self.H[i,j], self.H[j,i], self.V[i,j], self.V[j,i])
            best[i] = min_cost if min_cost != np.inf else 0
        return best

    def _heuristic(self, placed_mask):
        """h(n): Sum of best costs for unplaced pieces"""
        return sum(self.best_buddy_costs[i] for i in range(self.n) if not placed_mask[i])

    def solve(self):
        """
        Greedy Best-First Search (faster approximation of A*)
        Instead of exploring all states, we only keep the single best path.
        This trades optimality for speed and memory.
        """
        # Find best seed (highest variance piece)
        variances = [np.std(p) for p in self.pieces_lab]
        seed_id = int(np.argmax(variances))
        
        # State: grid as flat array, placed mask
        grid = np.full(self.n, -1, dtype=np.int32)  # grid[row*N + col] = piece_id
        grid[0] = seed_id  # Place seed at position 0 (top-left)
        
        placed = np.zeros(self.n, dtype=bool)
        placed[seed_id] = True
        
        g_cost = 0  # Running cost
        
        # Greedy placement loop
        for step in range(1, self.n):
            best_move = None
            best_f = np.inf
            
            # Find all empty slots adjacent to filled ones
            for idx in range(self.n):
                if grid[idx] != -1: continue  # Already filled
                
                r, c = idx // PUZZLE_N, idx % PUZZLE_N
                
                # Check if adjacent to any placed piece
                has_neighbor = False
                if c > 0 and grid[idx-1] != -1: has_neighbor = True
                if c < PUZZLE_N-1 and grid[idx+1] != -1: has_neighbor = True
                if r > 0 and grid[idx-PUZZLE_N] != -1: has_neighbor = True
                if r < PUZZLE_N-1 and grid[idx+PUZZLE_N] != -1: has_neighbor = True
                
                if not has_neighbor: continue
                
                # Try all unplaced pieces
                for pid in range(self.n):
                    if placed[pid]: continue
                    
                    # Calculate g(n) - cost of this placement
                    local_cost = 0
                    num_edges = 0
                    
                    if c > 0 and grid[idx-1] != -1:  # Left neighbor
                        local_cost += self.H[grid[idx-1], pid]
                        num_edges += 1
                    if c < PUZZLE_N-1 and grid[idx+1] != -1:  # Right neighbor
                        local_cost += self.H[pid, grid[idx+1]]
                        num_edges += 1
                    if r > 0 and grid[idx-PUZZLE_N] != -1:  # Top neighbor
                        local_cost += self.V[grid[idx-PUZZLE_N], pid]
                        num_edges += 1
                    if r < PUZZLE_N-1 and grid[idx+PUZZLE_N] != -1:  # Bottom neighbor
                        local_cost += self.V[pid, grid[idx+PUZZLE_N]]
                        num_edges += 1
                    
                    if num_edges == 0: continue
                    
                    avg_cost = local_cost / num_edges
                    
                    # Calculate h(n) - heuristic for remaining pieces
                    placed[pid] = True
                    h_cost = self._heuristic(placed)
                    placed[pid] = False
                    
                    # f(n) = g(n) + h(n)
                    f_cost = (g_cost + avg_cost) + h_cost
                    
                    if f_cost < best_f:
                        best_f = f_cost
                        best_move = (idx, pid, avg_cost)
            
            # Execute best move
            if best_move:
                idx, pid, cost = best_move
                grid[idx] = pid
                placed[pid] = True
                g_cost += cost
            else:
                # Fallback: fill remaining slots with remaining pieces
                remaining = [i for i in range(self.n) if not placed[i]]
                for idx in range(self.n):
                    if grid[idx] == -1 and remaining:
                        grid[idx] = remaining.pop(0)
                break
        
        # Reshape to 2D
        final_grid = grid.reshape(PUZZLE_N, PUZZLE_N)
        
        # Reconstruct image
        rows = []
        for r in range(PUZZLE_N):
            rows.append(np.hstack([self.pieces_bgr[final_grid[r,c]] for c in range(PUZZLE_N)]))
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
    print(f"--- EFFICIENT A* SOLVER: {len(puzzle_files)} 8x8 PUZZLES ---")
    print(f"{'Image':<30} | {'Accuracy':<10}")
    print("-" * 45)

    total_acc, count = 0, 0

    for p_file in puzzle_files:
        try:
            solver = EfficientAStarSolver8x8(os.path.join(PUZZLE_FOLDER, p_file))
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