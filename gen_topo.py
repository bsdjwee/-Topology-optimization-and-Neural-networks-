import os

# ⭐ 必须在所有科学计算库 import 之前
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import random
import time
import warnings
import concurrent.futures
import multiprocessing

# 屏蔽非致命数值警告
np.seterr(all='ignore')
warnings.filterwarnings('ignore')

# ==========================================
# 1. 核心求解器 (SIMP Robust Version)
# ==========================================
class TopOptSolver:
    def __init__(self, nelx, nely, volfrac, penal, rmin):
        self.nelx = nelx
        self.nely = nely
        self.volfrac = volfrac
        self.penal = penal
        self.rmin = rmin
        self.ndof = 2 * (nelx + 1) * (nely + 1)
        self.init_fe_matrices()

    def init_fe_matrices(self):
        E0, nu = 1.0, 0.3
        k = np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
        self.KE = E0/(1-nu**2)*np.array([
            [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
            [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
            [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
            [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
            [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
            [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
            [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
            [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]
        ])
        
        edofMat = np.zeros((self.nelx * self.nely, 8), dtype=int)
        for elx in range(self.nelx):
            for ely in range(self.nely):
                el = ely + elx * self.nely
                n1 = (self.nely + 1) * elx + ely
                n2 = (self.nely + 1) * (elx + 1) + ely
                edofMat[el, :] = np.array([2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3, 2*n2, 2*n2+1, 2*n1, 2*n1+1])
        
        self.edofMat = edofMat
        self.iK = np.kron(edofMat, np.ones((8, 1))).flatten()
        self.jK = np.kron(edofMat, np.ones((1, 8))).flatten()
        
        self.H, self.Hs = self.prepare_filter()

    def prepare_filter(self):
        nfilter = int(self.nelx * self.nely * ((2 * (np.ceil(self.rmin) - 1) + 1)**2))
        iH = np.zeros(nfilter)
        jH = np.zeros(nfilter)
        sH = np.zeros(nfilter)
        cc = 0
        for i in range(self.nelx):
            for j in range(self.nely):
                row = i * self.nely + j
                kk1 = int(np.maximum(i - (np.ceil(self.rmin) - 1), 0))
                kk2 = int(np.minimum(i + np.ceil(self.rmin), self.nelx))
                ll1 = int(np.maximum(j - (np.ceil(self.rmin) - 1), 0))
                ll2 = int(np.minimum(j + np.ceil(self.rmin), self.nely))
                for k in range(kk1, kk2):
                    for l in range(ll1, ll2):
                        col = k * self.nely + l
                        fac = self.rmin - np.sqrt((i - k)**2 + (j - l)**2)
                        iH[cc] = row
                        jH[cc] = col
                        sH[cc] = np.maximum(0.0, fac)
                        cc += 1
        H = coo_matrix((sH, (iH, jH)), shape=(self.nelx*self.nely, self.nelx*self.nely)).tocsc()
        Hs = np.array(H.sum(1)).flatten()
        return H, Hs

    def optimality_criteria(self, x, dc, dv):
        l1, l2, move = 0, 100000, 0.2
        # 数值稳定处理：防止开根号出现负数
        numerator = np.maximum(1e-10, -dc)
        while (l2 - l1) / (l1 + l2) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            # 防止除以零
            scaling = np.sqrt(numerator / (dv * lmid + 1e-9))
            xnew = np.maximum(0.001, np.maximum(x - move, np.minimum(1.0, np.minimum(x + move, x * scaling))))
            if np.sum(xnew) - self.volfrac * self.nelx * self.nely > 0:
                l1 = lmid
            else:
                l2 = lmid
        return xnew

    def solve(self, force_vector, fixed_dofs, max_loop=100, return_history=False):
        x = self.volfrac * np.ones(self.nely * self.nelx, dtype=float)
        free_dofs = np.setdiff1d(np.arange(self.ndof), fixed_dofs)
    
        if len(free_dofs) < self.ndof * 0.1:
            raise ValueError("Too many constraints")
    
        loop = 0
        change = 1.0
        u = np.zeros((self.ndof, 1))
    
        # ⭐ 新增：记录历史
        history = []
    
        while loop < max_loop:
            loop += 1
    
            # ⭐ 保存当前密度（reshape成2D）
            if return_history:
                history.append(x.reshape((self.nelx, self.nely)).copy())
    
            sK = ((self.KE.flatten()[np.newaxis]).T * (1e-9 + x**self.penal * (1 - 1e-9))).flatten(order='F')
            K = coo_matrix((sK, (self.iK, self.jK)), shape=(self.ndof, self.ndof)).tocsc()
    
            K_free = K[free_dofs, :][:, free_dofs]
            f_free = force_vector[free_dofs, 0]
            u[free_dofs, 0] = spsolve(K_free, f_free)
    
            u_e = u[self.edofMat].reshape(self.nelx * self.nely, 8)
            ce = np.sum(np.dot(u_e, self.KE) * u_e, axis=1)
    
            dc = -self.penal * x**(self.penal - 1) * ce
            dv = np.ones(self.nely * self.nelx)
            dc = np.asarray(self.H * (x * dc / self.Hs)) / np.maximum(0.001, x)
    
            xnew = self.optimality_criteria(x, dc, dv)
            change = np.linalg.norm(xnew - x, np.inf)
            x = xnew
    
        # ⭐ 最后一帧也存
        if return_history:
            history.append(x.reshape((self.nelx, self.nely)).copy())
            return x.reshape((self.nelx, self.nely)), np.array(history)

        return x.reshape((self.nelx, self.nely))

# ==========================================
# 2. 增强型工况生成器 (支持全向旋转)
# ==========================================
def generate_augmented_sample(nelx, nely):
    """
    20% 标准 (支持上下左右旋转) + 80% 随机
    """
    ndof = 2 * (nelx + 1) * (nely + 1)
    f = np.zeros((ndof, 1))
    fixed_dofs = []
    
    volfrac = random.uniform(0.4, 0.6)
    dice = random.random()
    case_type = "random"

    # 辅助函数：获取某一条边的所有节点
    def get_side_nodes(side_idx):
        # 0:Left, 1:Right, 2:Top, 3:Bottom
        nodes = []
        if side_idx == 0: # Left x=0
            for y in range(nely + 1): nodes.append(y)
        elif side_idx == 1: # Right x=nelx
            for y in range(nely + 1): nodes.append((nely+1)*nelx + y)
        elif side_idx == 2: # Top y=0
            for x in range(nelx + 1): nodes.append((nely+1)*x + 0)
        elif side_idx == 3: # Bottom y=nely
            for x in range(nelx + 1): nodes.append((nely+1)*x + nely)
        return nodes

    # --- Type 1: 旋转悬臂梁 (Cantilever) ~ 5% ---
    if dice < 0.05:
        case_type = "cantilever"
        # 随机选择一个固定面 (0=左, 1=右, 2=上, 3=下)
        fix_side = random.randint(0, 3)
        
        # 固定该面所有节点
        fix_nodes = get_side_nodes(fix_side)
        for n in fix_nodes:
            fixed_dofs.extend([2*n, 2*n+1])
            
        # 在对面施加力
        # 对面索引: 0<->1, 2<->3
        load_side = [1, 0, 3, 2][fix_side]
        load_nodes = get_side_nodes(load_side)
        
        # 随机选一个受力点
        target_node = random.choice(load_nodes)
        
        # 力方向：垂直于悬臂方向 (模拟弯矩)
        if fix_side in [0, 1]: # 左右固定 -> 力是垂直的(Y方向)
            f[2*target_node+1, 0] = random.choice([-1.0, 1.0])
        else: # 上下固定 -> 力是水平的(X方向)
            f[2*target_node, 0] = random.choice([-1.0, 1.0])

    # --- Type 2: 旋转简支梁 (Bridge) ~ 5% ---
    elif dice < 0.1:
        case_type = "bridge"
        # 随机选择“地面” (0=左, 1=右, 2=上, 3=下)
        ground_side = random.randint(0, 3)
        g_nodes = get_side_nodes(ground_side)
        
        # 桥墩宽度
        pier_width = 5
        
        # 固定地面两端的桥墩
        # g_nodes 是排好序的，直接取头尾即可
        pier1 = g_nodes[:pier_width]
        pier2 = g_nodes[-pier_width:]
        
        for n in pier1 + pier2:
            fixed_dofs.extend([2*n, 2*n+1]) # 强约束
            
        # 在对面中间施加力
        load_side = [1, 0, 3, 2][ground_side]
        l_nodes = get_side_nodes(load_side)
        
        # 取中间区域
        mid_idx = len(l_nodes) // 2
        # 在中间范围波动一下
        offset = random.randint(-int(len(l_nodes)*0.2), int(len(l_nodes)*0.2))
        target_node = l_nodes[mid_idx + offset]
        
        # 力方向：指向“地面”
        if ground_side == 0: # 地面在左 -> 力向左 (-X)
            f[2*target_node, 0] = -1.0
        elif ground_side == 1: # 地面在右 -> 力向右 (+X)
            f[2*target_node, 0] = 1.0
        elif ground_side == 2: # 地面在上 -> 力向上 (-Y)
            f[2*target_node+1, 0] = -1.0
        elif ground_side == 3: # 地面在下 -> 力向下 (+Y) [经典桥]
            f[2*target_node+1, 0] = 1.0 # 这里虽然向下通常是+Y(取决于坐标系定义)，SIMP里通常向下为正或负不影响拓扑，只要一致即可。这里假设坐标系y向下。

    # --- Type 3: 边界随机约束 (Random) ~ 90% ---
    else:
        case_type = "random"
        # 保持之前的随机逻辑
        num_constraints = random.randint(1, 3)
        for _ in range(num_constraints):
            side = random.randint(0, 3)
            seg_len = random.randint(5, 20)
            nodes = get_side_nodes(side)
            
            # 在该边上随机选一段
            start_idx = random.randint(0, len(nodes) - seg_len)
            seg_nodes = nodes[start_idx : start_idx + seg_len]
            
            for n in seg_nodes:
                fixed_dofs.extend([2*n, 2*n+1])

        # 随机力 (边界)
        num_forces = random.randint(1, 3)
        
        # 预先提取固定点坐标，用于距离检查
        fixed_node_indices = set([d // 2 for d in fixed_dofs])
        fixed_coords = []
        for n_idx in fixed_node_indices:
            cx = n_idx // (nely + 1)
            cy = n_idx % (nely + 1)
            fixed_coords.append((cx, cy))
        
        min_dist_sq = 12**2  # 距离阈值平方 (dist=12)

        for _ in range(num_forces):
            # 尝试多次寻找合适位置
            for _try in range(50):
                # 随机选择一条边: 0=Left, 1=Right, 2=Top, 3=Bottom
                force_side = random.randint(0, 3)
                
                if force_side == 0: # Left x=0
                    fx = 0
                    fy = random.randint(0, nely)
                elif force_side == 1: # Right x=nelx
                    fx = nelx
                    fy = random.randint(0, nely)
                elif force_side == 2: # Top y=0
                    fx = random.randint(0, nelx)
                    fy = 0
                else: # Bottom y=nely
                    fx = random.randint(0, nelx)
                    fy = nely
                
                # 检查与所有固定点的距离
                too_close = False
                for (cx, cy) in fixed_coords:
                    if (fx - cx)**2 + (fy - cy)**2 < min_dist_sq:
                        too_close = True
                        break
                
                if not too_close:
                    # 位置合法，添加力并跳出重试循环
                    node_idx = (nely + 1) * fx + fy
                    f[2*node_idx, 0] += random.uniform(-1, 1)
                    f[2*node_idx+1, 0] += random.uniform(-1, 1)
                    break

    fixed_dofs = np.array(list(set(fixed_dofs)))
    return volfrac, f, fixed_dofs, case_type

# ==========================================
# 3. 主程序
# ==========================================
def worker_task(idx, nelx, nely, output_dir, save_npz, timestamp_str):
    """
    单个进程的工作函数：不断尝试直到生成一个有效样本
    """
    # 确保每个进程有独立的随机种子
    import time
    np.random.seed(int(time.time() * 1000000 + idx) % 2**32)
    random.seed()
    
    while True:
        try:
            volfrac, f, fixed_dofs, c_type = generate_augmented_sample(nelx, nely)
            
            # 基础检查
            if len(fixed_dofs) < 5 or np.sum(np.abs(f)) < 1e-6:
                continue
                
            rmin_val = random.uniform(2.2, 3.0)
            solver = TopOptSolver(nelx=nelx, nely=nely, volfrac=volfrac, penal=3.0, rmin=rmin_val)
            x_structure, history = solver.solve(f, fixed_dofs, max_loop=100, return_history=True)
            
            if np.mean(x_structure) < 0.05 or np.mean(x_structure) > 0.95:
                continue # Diverged (or empty/full)

            filename = os.path.join(output_dir, f"{timestamp_str}_{idx:06d}")
            plt.imsave(f"{filename}.png", 1 - x_structure.T, cmap='gray')
            
            if save_npz:
                np.savez(
                    f"{filename}.npz",
                    structure=x_structure,
                    history=history,           # ⭐ 新增
                    force=f,
                    fixed_dofs=fixed_dofs,
                    volfrac=volfrac,
                    type=c_type
                )
            
            return c_type
        except Exception:
            pass

if __name__ == "__main__":
    # 配置是否保存 .npz 文件
    SAVE_NPZ = True

    NELX, NELY = 64, 64
    TARGET_SAMPLES = 40000
    OUTPUT_DIR = "output_topo_40000" # [修改] 输出文件夹改名
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # 固定为 16 个核
    MAX_WORKERS = 12
    
    # 生成时间戳前缀 (日时分)
    import datetime
    timestamp_str = datetime.datetime.now().strftime("%d%H%M")
    
    print(f"=== Generating Augmented Dataset ({NELX}x{NELY}) ===")
    print(f"Rule: 10% Rotated Standard (Cantilever/Bridge), 90% Random")
    print(f"Saving to: {OUTPUT_DIR}")
    print(f"File prefix: {timestamp_str}_xxx")
    print(f"Parallel processing with {MAX_WORKERS} workers")
    
    valid_count = 0
    t_start = time.time()
    stats = {'cantilever': 0, 'bridge': 0, 'random': 0}
    
    # 使用 ProcessPoolExecutor 进行并行计算
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        futures = []
        for i in range(TARGET_SAMPLES):
            futures.append(
                executor.submit(worker_task, i, NELX, NELY, OUTPUT_DIR, SAVE_NPZ, timestamp_str)
            )
        # 获取结果（按完成顺序）
        for future in concurrent.futures.as_completed(futures):
            try:
                c_type = future.result()
                valid_count += 1
                stats[c_type] += 1
                
                if valid_count % 10 == 0 or valid_count == TARGET_SAMPLES:
                    elapsed = time.time() - t_start
                    speed = valid_count / elapsed
                    print(f"[OK] {valid_count}/{TARGET_SAMPLES} | Type: {c_type:10s} | Time: {elapsed:.1f}s | Speed: {speed:.2f} it/s")
            except Exception as exc:
                print(f"Task generated an exception: {exc}")

    print(f"\nDone! Stats: {stats}")