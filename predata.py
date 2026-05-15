# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
from tqdm import tqdm


# ===============================
# DOF → 节点网格
# ===============================

def dof_to_node_grid(fixed_dofs, force, nelx, nely):
    """
    将自由度表示的边界条件和力转换为节点网格
    fixed_dofs: 固定自由度的索引数组
    force: 力向量 (ndof, 1)
    nelx, nely: 单元数
    返回: bc(2, nely+1, nelx+1), load(2, nely+1, nelx+1)
    """
    bc = np.zeros((2, nely + 1, nelx + 1), dtype=np.float32)
    load = np.zeros((2, nely + 1, nelx + 1), dtype=np.float32)

    # 固定边界
    for dof in fixed_dofs:
        node = dof // 2
        direction = dof % 2

        y = node // (nelx + 1)  # 行索引
        x = node % (nelx + 1)   # 列索引

        bc[direction, y, x] = 1.0

    # 力
    force_flat = force.flatten()

    for dof, f_val in enumerate(force.flatten()):
        if abs(f_val) > 1e-12:
            node = dof // 2
            direction = dof % 2
    
            y = node // (nelx + 1)
            x = node % (nelx + 1)
    
            load[direction, y, x] = f_val
    return bc, load
    
def element_to_node_physical(structure):
    """
    将单元网格转换为节点网格
    structure: (H, W) = (nelx, nely) 单元网格
    return: (1, H+1, W+1) 节点网格
    """
    if isinstance(structure, np.ndarray):
        structure = torch.from_numpy(structure).float()
    
    H, W = structure.shape
    
    # 初始化节点网格
    node_grid = torch.zeros((H+1, W+1), dtype=torch.float32)
    
    # 计算四个角点的贡献
    node_grid[:-1, :-1] += structure  # 左上
    node_grid[1:, :-1] += structure  # 左下
    node_grid[:-1, 1:] += structure  # 右上
    node_grid[1:, 1:] += structure   # 右下
    
    # 归一化
    count = torch.zeros((H+1, W+1), dtype=torch.float32)
    count[:-1, :-1] += 1
    count[1:, :-1] += 1
    count[:-1, 1:] += 1
    count[1:, 1:] += 1
    
    node_grid = node_grid / count
    
    return node_grid.unsqueeze(0)  # (1, H+1, W+1)


# ===============================
# 主预处理函数
# ===============================

def preprocess_dataset(
        raw_dir="/home/hym001/diffusion/data_self_surpport/output_topo_40000/",
        save_dir="/home/hym001/MLforTOP/ml/data/processed_support1/",
        print_info=True
):

    """
    预处理数据集，将所有物理场转换为节点网格(65×65)
    并保留密度场的连续值
    
    参数:
        raw_dir: 原始npz数据目录
        save_dir: 预处理后的pt文件保存目录
        print_info: 是否打印详细信息
    """
    os.makedirs(save_dir, exist_ok=True)

    files = sorted(
        [f for f in os.listdir(raw_dir) if f.endswith(".npz")]
    )

    if len(files) == 0:
        print("没有找到 .npz 文件")
        return

    print(f"发现 {len(files)} 个 npz 文件")
    print(f"原始数据目录: {raw_dir}")
    print(f"输出目录: {save_dir}")
    
    for idx, fname in enumerate(tqdm(files, desc="处理文件")):
        try:
            # 加载数据
            data_path = os.path.join(raw_dir, fname)
            data = np.load(data_path)
            
            # 必需字段
            structure = data["structure"]  # (64, 64) 单元网格
            force = data["force"]          # (ndof, 1)
            fixed_dofs = data["fixed_dofs"]  # 固定自由度索引
            
            # 获取体积分数，默认为0.4
            if "volfrac" in data:
                volfrac = float(data["volfrac"])
            else:
                volfrac = 0.4
            
            nelx, nely = structure.shape
            
            # ===============================
            # 边界条件和载荷 → 节点网格
            # ===============================
            bc, load = dof_to_node_grid(fixed_dofs, force, nelx, nely)
            
            # ===============================
            # 结构密度场 → 节点网格
            # ===============================
            structure_node = element_to_node_physical(structure)  # (1, 65, 65)
            
            # 转换为torch张量
            bc_tensor = torch.from_numpy(bc).float()        # (2, 65, 65)
            load_tensor = torch.from_numpy(load).float()    # (2, 65, 65)
            
            # ===============================
            # 构建样本（只保存5个字段）
            # ===============================
            sample = {
                "structure": structure_node,        # 密度场 (1, 65, 65)，连续值0-1
                "bc": bc_tensor,                   # 边界条件 (2, 65, 65)，xy方向固定
                "load": load_tensor,               # 载荷 (2, 65, 65)，xy方向力
                "volfrac": torch.tensor(volfrac, dtype=torch.float32),  # 体积分数
                "file_name": fname                  # 原始文件名
            }
            
            # 保存
            
            start_idx = 20004  # 从20004开始编号
            save_path = os.path.join(save_dir, f"sample_{start_idx + idx:06d}.pt")
            torch.save(sample, save_path)
            
            if print_info and idx < 3:  # 只打印前3个文件的信息
                print(f"\n文件: {fname}")
                print(f"  结构形状: {structure.shape} -> {structure_node.shape}")
                print(f"  BC形状: {bc.shape}")
                print(f"  载荷形状: {load.shape}")
                print(f"  体积分数: {volfrac:.3f}")
                print(f"  平均密度: {torch.mean(structure_node).item():.3f}")
                print(f"  固定自由度数: {len(fixed_dofs)}")
                
        except Exception as e:
            print(f"\n处理 {fname} 出错: {e}")
            continue
    
    # 打印汇总信息
    print(f"\n预处理完成，共生成 {len(files)} 个样本")
    print(f"保存目录: {save_dir}")
    
    # 验证第一个样本
    if len(files) > 0:
        sample = torch.load(os.path.join(save_dir, "sample_000000.pt"))
        print("\n第一个样本的字段:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: {type(value).__name__} = {value}")


# ===============================
# 主程序
# ===============================
if __name__ == "__main__":
    # 使用默认路径直接运行
    preprocess_dataset()