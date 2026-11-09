import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import math
from collections import Counter
from typing import List, Tuple, Any, Dict

from rdkit import Chem
from rdkit.Chem import Draw, rdmolops, AllChem
import re
import sys
import os

from element_to_AtomicNumber import get_AtomicNumber, element_to_AtomicNumber

_GROVE_DIR = os.path.dirname(os.path.abspath(__file__))

# Cache atomic symbols list to avoid repeated conversions
_ATOMIC_SYMBOLS = list(element_to_AtomicNumber.keys())
_ATOMIC_SYMBOLS_SET = set(_ATOMIC_SYMBOLS)

# 定义一个函数用于筛选出最大包含项
def filter_max_included_matches(fragments):
    # Pre-compute match sets to avoid repeated conversions
    fragments_with_sets = [(frag, set(frag[1]), frag[1]) for frag in fragments]
    filtered_fragments = []
    
    for idx, (new_group, new_match_set, new_group_1) in enumerate(fragments_with_sets):
        to_add = True
        
        # 遍历所有的 fragments，判断是否有包含关系
        for existing_idx, (existing_group, existing_match_set, existing_group_1) in enumerate(fragments_with_sets):
            if idx == existing_idx:
                continue # 不与自己比较
            
            # 如果当前匹配集是其他项的子集，则标记为要移除
            if new_match_set.issubset(existing_match_set) and new_match_set != existing_match_set:
                to_add = False
                break

            # 如果有两个完全相同、顺序不同的，保留从小到大排列的
            if new_match_set == existing_match_set:
                # 查找第一个不同的元素位置
                for i in range(len(new_group_1)):
                    if new_group_1[i] != existing_group_1[i]:
                        if new_group_1[i] > existing_group_1[i]:
                            to_add = False
                        break

    
        # 如果没有被任何其他项包含，保留该项
        if to_add:
            filtered_fragments.append(new_group)
    
    # 对结果按匹配位置进行排序
    filtered_fragments.sort(key=lambda x: x[1])
    
    return filtered_fragments

# 提取所有官能团或原子团
def extract_FunGroup(smiles, functional_groups):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    fragments = []
    seen = set()  # 储存已处理的 (smarts, match) 组合

    # 使用 GetSymmSSSR 获取所有环
    rings = [list(ring) for ring in Chem.GetSymmSSSR(mol)]
    for ring in rings:
        ring = sorted(ring)
        fragments.append(('Ring', tuple(ring), 'Ring'))

    # 使用 SMARTS 进行官能团匹配
    for _, row in functional_groups.iterrows():
        atom_smarts = row[0]
        smarts = row[1]
        is_ring = row[2]
        pattern = Chem.MolFromSmarts(smarts)
        matches = mol.GetSubstructMatches(pattern)

        for match in matches:
            if (atom_smarts, match) not in seen:
                fragments.append((atom_smarts, match, is_ring))
                seen.add((atom_smarts, match))

    for idx_fragments, frag in enumerate(fragments):
        if frag[0] == 'Ring':
            # 获取环的索引
            indices = frag[1]

            # 默认是烷环（C为大写）
            atoms = ['C'] * len(indices)

            # 直接检查每个原子是否为芳香原子，避免 PathToSubmol 误用原子索引作键索引
            all_aromatic = all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in indices)
            if all_aromatic:  # 芳香环
                atoms = [atom.lower() for atom in atoms]
            
            # 遍历每个环中的原子索引
            # 预计算非环fragments的索引映射，避免重复查找
            non_ring_fragments = fragments[len(rings):]
            atom_index_to_symbol = {}
            for fragment in non_ring_fragments:
                if any(atom_index in fragment[1] for atom_index in frag[1]):
                    atom_result = process_atoms_with_indices(fragment)
                    for atom_symbol, idx in atom_result:
                        if idx in frag[1] and atom_symbol not in {'C', 'c'}:
                            atom_index_to_symbol[idx] = atom_symbol
            
            for (idx_ring, atom_index) in enumerate(frag[1]):
                if atom_index in atom_index_to_symbol:
                    atoms[idx_ring] = atom_index_to_symbol[atom_index]

            # 更新 frag
            atoms[0] += '1' # 在第一个原子后面加上 1
            atoms[-1] += '1' # 在最后一个原子后面加上 1
            atoms_str = ''.join(atoms)
            fragments[idx_fragments] = (atoms_str, frag[1], frag[2])

    # 删掉与前 len(rings) 行索引重复的行
    # 提取前 len(rings) 行的第二列，使用set提高查找效率
    refer_sets = {fragment[1] for fragment in fragments[:len(rings)]}

    filtered_fragments = fragments[:len(rings)]
    for fragment in fragments[len(rings):]:
        # 判断当前行的第二列是否在前 len(rings) 的第二列中
        if fragment[1] in refer_sets:
            continue
        filtered_fragments.append(fragment)

    # 按照匹配位置从小到大排序
    filtered_fragments.sort(key=lambda x: x[1])
    
    # 过滤掉包含关系较小的项，只保留最大包含项
    filtered_fragments = filter_max_included_matches(filtered_fragments)

    filtered_fragments_updated = []
    for idx_fragments, frag in enumerate(filtered_fragments):
        frag_second_col = frag[1]
        if frag[2] == 'Ring':
            # 处理特殊情况 'C(=O)NC(=O)'
            if idx_fragments + 1 < len(filtered_fragments) and filtered_fragments[idx_fragments + 1][0] == 'C(=O)NC(=O)' and len(set(frag_second_col) & set(filtered_fragments[idx_fragments + 1][1])) >= 3:
                continue
        
        filtered_fragments_updated.append(frag)

    return filtered_fragments_updated

# 检测官能团种类及数量，并计算溶解度
def calc_delta_from_fragments(
        fragments: List[Tuple[str, Any, str]],
        weight_csv: str = None
) -> Dict[str, Any]:
    if weight_csv is None:
        weight_csv = os.path.join(_GROVE_DIR, "FunGroup_Weight.csv")
    df = pd.read_csv(weight_csv, index_col=0)
    weights = dict(
        zip(df.index.astype(str), df.iloc[:, 0].astype(float))
    )

    counts = Counter(frag[0] for frag in fragments)
    contributions = {
        fg: counts[fg] * weights[fg]
        for fg in counts
        if fg in weights
    }

    total_score = sum(contributions.values())
    return{
        "total_score": total_score,
        "counts": counts,
        "contributions": contributions
    }

# 检查数字序列是否连续
def is_continuous_sequence(sorted_nums):
    """检查排序后的数字序列是否连续"""
    if len(sorted_nums) <= 1:
        return True
    return all(sorted_nums[i] + 1 == sorted_nums[i + 1] for i in range(len(sorted_nums) - 1))

# 寻找最大连续度序号
def find_max_consecutive(nums):
    if not nums:
        return -1
    max_end = nums[0]
    current_end = nums[0]
    nums_len = len(nums)
    for i in range(1, nums_len):
        if nums[i] == nums[i - 1] + 1:
            current_end = nums[i]
        else:
            break

    return current_end # 最大连续段的终点

# 重新排序 fragments，按照序号、交集元素和连续度
def sort_fragments(fragments):
    # 1. 提取第二列元素集（用索引追踪，避免 frozenset 碰撞）
    fragments_sets = [set(f[1]) for f in fragments]

    # 2. 初始化排序索引列表和未排序集合
    sorted_indices = []
    remaining_rows = set(range(len(fragments_sets)))

    # 3. 先找出包含 0 的行作为起点
    current_set = None
    for idx in list(remaining_rows):
        if 0 in fragments_sets[idx]:
            current_set = fragments_sets[idx]
            sorted_indices.append(idx)
            remaining_rows.remove(idx)
            break

    if current_set is None and remaining_rows:
        # 没有含 0 的行，取第一行作起点
        idx = min(remaining_rows)
        current_set = fragments_sets[idx]
        sorted_indices.append(idx)
        remaining_rows.remove(idx)

    # 4. 依次选择下一行，确保所有行都被排序
    while remaining_rows:
        # 找与当前集合交集最大的行
        best_idx = None
        max_intersection = -1 # 允许 0 交集匹配

        # 在所有未排序的行中，寻找交集最大的行
        for idx in remaining_rows:
            row = fragments_sets[idx]
            intersection_size = len(current_set.intersection(row))
            if intersection_size > max_intersection:
                best_idx = idx
                max_intersection = intersection_size

        if best_idx is not None and max_intersection > 0:
            current_set = fragments_sets[best_idx]
            sorted_indices.append(best_idx)
            remaining_rows.remove(best_idx)
        else:
            # 计算当前已排序元素的所有数值
            merged_elements = sorted(set().union(*(fragments_sets[i] for i in sorted_indices)))

            # 找到最大连续度终点
            max_consecutive_end = find_max_consecutive(merged_elements)

            # 选择包含 max_consecutive_end + 1 的行继续排序
            next_element = max_consecutive_end + 1
            best_idx = None
            for idx in remaining_rows:
                if next_element in fragments_sets[idx]:
                    best_idx = idx
                    break # 只选取一个

            if best_idx is not None:
                current_set = fragments_sets[best_idx]
                sorted_indices.append(best_idx)
                remaining_rows.remove(best_idx)
            else:
                # 如果没有找到匹配的行，直接加入剩余的行
                sorted_indices.extend(sorted(remaining_rows))
                break

    # 5. 按排序后的索引还原到原始 'fragments' 格式
    return [fragments[i] for i in sorted_indices]

# 调整非环类在环类中的位置，不改变非环类内原始相对位置
def sort_fragments_ringclass(fragments):
    # 处理 'Ring' 行
    ring_fragments = [frag for frag in fragments if frag[2] == 'Ring']
    ring_positions = {num for frag in ring_fragments for num in frag[1]}

    # 处理 'Non-ring' 行
    non_ring_fragments = [frag for frag in fragments if frag[2] == 'Non-ring']

    # 保留第一个 Ring 片段前的所有 Non-ring 片段
    first_ring_index = next((i for i, frag in enumerate(fragments) if frag[2] == 'Ring'), len(fragments))
    preserved_non_ring = fragments[:first_ring_index]
    remaining_non_ring_fragments = [frag for frag in non_ring_fragments if frag not in preserved_non_ring]

    new_sorted_fragments = ring_fragments[:]

    # **顺序遍历 Non-ring，确保插入顺序不变**
    for non_ring in remaining_non_ring_fragments:
        min_non_ring = min(non_ring[1])
        inserted = False
        
        for decrement in range(min_non_ring - 1, -1, -1):  # 递减查找
            if decrement in ring_positions:
                for i in range(len(new_sorted_fragments)):  # **从上到下遍历，保证顺序**
                    k = len(new_sorted_fragments) - i - 1
                    if decrement in new_sorted_fragments[k][1]:
                        insert_idx = k + 1
                        while insert_idx < len(new_sorted_fragments) and new_sorted_fragments[insert_idx][2] == 'Non-ring':
                            insert_idx += 1  # 找到最后一个连续 Non-ring 之后的位置
                        new_sorted_fragments.insert(insert_idx, non_ring)
                        inserted = True
                        break
            if inserted:
                break
        
        if not inserted:
            new_sorted_fragments.append(non_ring)  # 若未找到合适位置，保留原顺序

    # **合并保留的 Non-ring**
    final_sorted_fragments = preserved_non_ring + new_sorted_fragments

    return final_sorted_fragments

################################## 处理所有带括号的官能团 ##################################
# 处理特殊情况：误认官能团为支链
def remove_FunGroup_bracket(smiles):
    origin_smiles = smiles
    # 替换
    smiles = smiles.replace("C(=O)", "CO")
    smiles = smiles.replace("C(=S)", "CS")

    return smiles
##########################################################################################

# 将所有括号分组
def merge_brackets(left, right):
    groups = []
    temp_left, temp_right = [], []

    i, j = 0, 0
    while i < len(left) or j < len(right):
        if i < len(left) and (j >= len(right) or left[i] < right[j]):
            temp_left.append(left[i])
            i += 1
        else:
            temp_right.append(right[j])
            j += 1
        
        if len(temp_left) == len(temp_right) and temp_left: # 当左右两组大小相同时，形成一个大组
            groups.append((temp_left, temp_right))
            temp_left, temp_right = [], []  # 重新初始化
    return groups

# 逐组处理括号级别关系
def extract_nested_pairs(left, right):
    merged = sorted(left + right) # 合并排序
    level_map = {} # 记录每个元素的层级
    pairs = [] # 存储嵌套对应关系

    left_set = set(left)
    left_stack = [] # 用栈来管理 left-right 关系

    for num in merged:
        if num in left_set:
            level = level_map[left_stack[-1]] + 1 if left_stack else 1
            level_map[num] = level
            left_stack.append(num)
        else: # right 元素
            if left_stack:
                last_left = left_stack.pop()
                pairs.append((last_left, num)) # 形成最近的 left-right 对应关系
                level_map[num] = level_map[last_left] # right 继承 left 的层级

    # 颠倒层级
    max_level = max(level_map.values())
    for key in level_map:
        level_map[key] = max_level - level_map[key] + 1

    return pairs, level_map

# 找到比目标值大的最小元素
def find_closest_greater(value, sorted_list):
    """在 sorted_list 中找到比 value 大的最小元素"""
    for num in sorted_list:
        if num > value:
            return num
    
    return None # 如果找不到更大的值

# 找到比目标值小的最大元素
def find_closest_smaller(value, sorted_list):
    """在 sorted_list 中找到比 value 小的最大元素"""
    for num in reversed(sorted_list):
        if num < value:
            return num
    
    return None # 如果找不到更小的值

# 检查 2 个数字在变量的同一行
def are_in_same_row(z, a1, a2):
    for row in z:
        if a1 in row and a2 in row:
            return True
    
    return False

# 检测环结束点后是否存在隐式支链 (非括号)
def detect_implicit_branch(smiles, right_idx):
    """
    只有在 )X) 或 )X... ) 这种结构下，X 才算隐式支链
    """
    idx = right_idx + 1
    branch_indices = []

    if idx >= len(smiles):
        return False, []

    # 1️⃣ 吃 X（官能团 / 连续原子）
    while idx < len(smiles):
        c = smiles[idx]

        if c.isalpha() or c in "=#":
            branch_indices.append(idx)
            idx += 1
            continue

        # ❌ 碰到环编号，直接否定
        if c.isdigit() or c == "%":
            return False, []

        break

    if not branch_indices:
        return False, []

    # 2️⃣ 关键判定：X 后面在当前层级内必须还能遇到 ')'
    depth = 0
    scan_idx = right_idx + 1

    while scan_idx < len(smiles):
        if smiles[scan_idx] == "(":
            depth += 1
        elif smiles[scan_idx] == ")":
            if depth == 0:
                # ✅ 说明 X 是被“夹在括号闭合前”的
                return True, branch_indices
            depth -= 1
        scan_idx += 1

    # ❌ 没有再遇到 ')'
    return False, []

    
# 添加 bracket 信息
def process_pairs_with_bracket(ring_systems, smiles, pairs, result, fragments, level_map, group_idx):
    """
    处理括号支链，同时检测隐式支链
    """
    result_third_col = sorted([row[2] for row in result]) # 提取 result 的第三列，并排序

    # 遍历 pairs，找到比 left 大的最小元素，比 right 小的最大元素
    mapped_pairs = []
    pairs_idx = 0
    pairs_idx_remove = []

    for left, right in pairs:
        left_closest = find_closest_greater(left, result_third_col)
        right_closest = find_closest_smaller(right, result_third_col)

        left_row_outer = next((row for row in result if row[2] == left - 1), None)
        right_row_outer = next((row for row in result if row[2] == right + 1), None)

        if left_closest is not None and right_closest is not None:
            left_row = next(row for row in result if row[2] == left_closest) # 找到对应的第二列元素
            right_row = next(row for row in result if row[2] == right_closest)
            mapped_pairs.append((left_row[1], right_row[1])) # 形成 (value1, value2)

            if left_row_outer and right_row_outer:
                if are_in_same_row(ring_systems, left_row_outer[1], right_row_outer[1]) and left_row[1] != right_row[1]:
                    for ele in range(left_row_outer[1] + 1, right_row_outer[1]): # 如果 left_row 和 right_row 内部也有一个环上原子
                        if any(ele in row for row in ring_systems):
                            pairs_idx_remove.extend([pairs_idx])
                            break
            pairs_idx += 1
    updated_fragments = []

    for idx, fragment in enumerate(fragments): # 更新 fragments 的第四列
        if fragment[3] != 'Non-bracket': # 若已经有 bracket 信息，则保留
            updated_fragments.append((fragment[0], fragment[1], fragment[2], fragment[3]))
            continue

        updated = False # 检查该行是否在 mapped_pairs 中
        for mapped_idx, (left, right) in enumerate(mapped_pairs):
            if left in fragment[1]:             
                pairs_left, pairs_right = pairs[mapped_idx] # 查找在 pairs 中的对应位置
                level_left = level_map.get(pairs_left, "N/A") # 获取该行所有元素的层级信息
                level_right = level_map.get(pairs_right, "N/A")
                level_info_str = (group_idx + 1, [(left, level_left), (right, level_right)]) # 构建层级信息字符串

                if mapped_idx in pairs_idx_remove: # 并且内部也要有一个环上原子
                    continue
                updated_fragments.append((fragment[0], fragment[1], fragment[2], level_info_str)) # 更新该行的第四列
                updated = True
                break # 一旦更新该行，跳出循环
        if not updated:
            updated_fragments.append((fragment[0], fragment[1], fragment[2], "Non-bracket")) # 如果该行没有被更新，设置为 "Non-bracket"
    
    # --- 隐式支链处理（修正版） ---
    for mapped_idx, (left_atom, right_atom) in enumerate(mapped_pairs):
        pairs_left, pairs_right = pairs[mapped_idx]

        exists, branch_indices = detect_implicit_branch(smiles, pairs_right)
        if not exists:
            continue

        for i, fragment in enumerate(updated_fragments):
            if fragment[3] != 'Non-bracket':
                continue

            if any(idx in fragment[1] for idx in branch_indices):
                level_left = level_map.get(pairs_left, "N/A")
                level_right = level_map.get(pairs_right, "N/A")
                level_info = (group_idx + 1,
                            [(pairs_left, level_left),
                            (pairs_right, level_right)])

                updated_fragments[i] = (
                    fragment[0],
                    fragment[1],
                    fragment[2],
                    level_info
                )

    return updated_fragments

# 检查官能团是否在括号内
def check_brackets(smiles, fragments, ring_systems):
    """
    原 check_brackets 逻辑保留，结合 process_pairs_with_bracket 改造
    """
    smiles = remove_FunGroup_bracket(smiles) # 处理特殊情况
    left_bracket_idx = [] # 存储 '(' 索引
    right_bracket_idx = [] # 存储 ')' 索引
    result = process_atoms_with_indices_smiles(smiles) # smiles 中原子索引

    for i, elem in enumerate(smiles):  # 去除所有括号并记录索引映射
        if elem in "(":
            left_bracket_idx.append(i) # '(' 索引
        elif elem in ")":
            right_bracket_idx.append(i) # ')' 索引

    group_brackets = merge_brackets(left_bracket_idx, right_bracket_idx) # 括号分组
    fragments = [tuple(list(fragment[:3]) + ['Non-bracket']) for fragment in fragments] # 把 fragments 第四列统一初始化为 'Non-bracket'

    if group_brackets: # 逐组处理
        all_pairs = []
        all_level_map = []

        for idx, (left_bracket_idx, right_bracket_idx) in enumerate(group_brackets):
            pairs, level_map = extract_nested_pairs(left_bracket_idx, right_bracket_idx)
            sorted_pairs = sorted(pairs, key=lambda x: x[0]) # 对每个 pairs 内部按第一个元素排序
            all_pairs.append(sorted_pairs)
            all_level_map.append(level_map)
        sorted_all_pairs_and_levels = sorted(zip(all_pairs, all_level_map), key=lambda x: x[0][0][0] if x[0] else float('inf')) # 根据每个 pairs 内的最小元素（第一个元素）排序
        if sorted_all_pairs_and_levels:
            sorted_all_pairs, sorted_all_level_map = zip(*sorted_all_pairs_and_levels) # 解压排序后的 pairs 和 level_map

            for group_idx, (pairs, level_map) in enumerate(zip(sorted_all_pairs, sorted_all_level_map)): # 按照 pairs 更新 fragments
                fragments = process_pairs_with_bracket(ring_systems, smiles, pairs, result, fragments, level_map, group_idx)
    return fragments
        
# 处理 smiles 转化时的杂环 (换原子版)
def process_hetero_exchange(bracket_frag, hetero_pos):
    result = []
    num = 1

    for i, current in enumerate(bracket_frag):
        # 查询环
        frag, frag_pos, is_ring, bracket_type = current
        ring_number = num # 环号

        # 筛选此环号的杂原子位置
        hetero_atoms_in_ring = [atom for atom in hetero_pos if atom[0] == ring_number]

        if is_ring == 'Ring' and hetero_atoms_in_ring:
            mol_frag = Chem.MolFromSmiles(frag)
            ring_info = mol_frag.GetRingInfo()
            ring_systems = ring_info.AtomRings()

            # 获得此环中的原子
            ring = ring_systems[0]

            # 记录当前环的原子
            ring_atoms = [mol_frag.GetAtomWithIdx(idx).GetSymbol() for idx in ring]
            # 重新排序
            ring, ring_atoms = sort_ring_with_atoms(ring, ring_atoms)

            # 把碳化后的原子替换回杂原子
            frag = list(frag) # 将 smiles 转为字符列表

            for pos in hetero_atoms_in_ring:
                element = pos[2]

                # 用元素符号来查找原子序号                
                atomic_number = get_AtomicNumber(element)

                # 替换回杂原子
                mol_frag.GetAtomWithIdx(ring[pos[1] - 1]).SetAtomicNum(atomic_number)

            updated_frag = Chem.MolToSmiles(mol_frag)

            # 用更新过的 frag 替换此行第一个元素
            bracket_frag[i] = list(bracket_frag[i]) # 将元组
            bracket_frag[i][0] = updated_frag

            num += 1

    return bracket_frag

# 处理 smiles 转化时的杂环 (**表述版)
def process_hetero(bracket_frag, hetero_pos):
    num = 0
    hetero_ring = [item[0] for item in hetero_pos]

    for i, current in enumerate(bracket_frag):
        # 查询环
        frag, frag_pos, is_ring, bracket_type = current
        frag = [frag] # 把字符串转化为列表
        if is_ring == 'Ring':
            num += 1

        # 筛选此环号的杂原子位置
        if is_ring == 'Ring' and num in hetero_ring:
            hetero_atoms_in_ring = [ele for ele in hetero_pos if ele[0] == num]

            for pos in hetero_atoms_in_ring:
                atom_pos = pos[1]
                element = pos[2] # 元素符号小写

                addition = f"*{atom_pos}{element}*"
                frag.append(addition)

            updated_frag = ''.join(frag)

            # 用更新过的 frag 替换此行第一个元素
            bracket_frag[i] = list(bracket_frag[i]) # 将元组转换为列表
            bracket_frag[i][0] = updated_frag
            bracket_frag[i] = tuple(bracket_frag[i]) # 将列表转换回元组

    return bracket_frag

# 处理碳化后的同位素、特殊表述
def process_isotope(hetero_frag, isotope_pos):
    for frag_idx, frag in enumerate(hetero_frag):
        for ele in isotope_pos:
            if ele[0] in frag[1]:
                atom_result = process_atoms_with_indices(frag)
                index_matches = [i for i, (_, value) in enumerate(atom_result) if value == ele[0]] # 获取是第几个原子
                if not index_matches:
                    continue
                index = index_matches[0]
                frag_smiles_result = process_atoms_with_indices_smiles(frag[0])
                symbol_matches = [row[2] for row in frag_smiles_result if row[1] == index]
                if not symbol_matches:
                    continue
                symbol_idx = symbol_matches[0]
                frag_smiles_list = list(frag[0])
                frag_smiles_list[symbol_idx] = ele[1]
                frag = list(frag)
                frag[0] = ''.join(frag_smiles_list)
                frag = tuple(frag)
                hetero_frag[frag_idx] = frag

                break

    return hetero_frag

# 提取 smiles 的原子索引
def process_atoms_with_indices_smiles(smiles):
    atoms = []
    tag = 0
    i = 0 # 当前字符的位置
    smiles_len = len(smiles)
    
    while i < smiles_len:
        # 尝试查找两个字符的原子符号
        if i + 1 < smiles_len and smiles[i:i+2] in _ATOMIC_SYMBOLS_SET:
            atoms.append((smiles[i:i+2], tag, i))
            tag += 1
            i += 2 # 跳过两个字符
        elif smiles[i].capitalize() in _ATOMIC_SYMBOLS_SET:
            atoms.append((smiles[i], tag, i))
            tag += 1
            i += 1 # 跳过一个字符
        else:
            i += 1 # 如果不是原子符号，跳过该字符

    return atoms

# 将 frag 里的原子符号与数字对应
def process_atoms_with_indices(frag):
    atom_symbols_str = frag[0]
    indices = frag[1]
    
    # 使用 process_atoms_with_indices_smiles 来解析原子符号
    atoms_info = process_atoms_with_indices_smiles(atom_symbols_str)
    
    # 将第二列的数字与当前符号的原子位置对应
    result = [(atom_info[0], indices[i]) for i, atom_info in enumerate(atoms_info) if i < len(indices)]
    
    return result

# 将 frag 里的原子符号与数字对应 (后处理)
def process_atoms_with_indices_after(frag):
    atom_symbols_str = frag[0]
    atom_symbol_idx = atom_symbols_str.find('*') # 找到第一个 '*' 的索引
    if atom_symbol_idx == -1:
        atom_symbol_idx = len(atom_symbols_str)
    
    # 只处理 '*' 之前的部分
    truncated_str = atom_symbols_str[:atom_symbol_idx]
    indices = frag[1]

    # 使用 process_atoms_with_indices_smiles 来解析原子符号
    atoms_info = process_atoms_with_indices_smiles(truncated_str)
    
    # 将第二列的数字与当前符号的原子位置对应
    result = [(atom_info[0], indices[i]) for i, atom_info in enumerate(atoms_info) if i < len(indices)]

    return result

# 计算元素中最大的相邻差值，并确定范围
def find_largest_diff_range(numbers):
    sorted_numbers = sorted(numbers)
    max_gap, target_end = 0, sorted_numbers[0]

    for i in range(len(sorted_numbers) - 1):
        gap = sorted_numbers[i + 1] - sorted_numbers[i]
        if gap > max_gap:
            max_gap = gap
            target_end = sorted_numbers[i]

    return sorted_numbers[0], target_end

# 计算元素中最大的相邻差值，并确定范围（从后往前）
def find_second_range(numbers):
    sorted_numbers = sorted(numbers, reverse=True)
    max_gap, target_start = 0, sorted_numbers[0]

    for i in range(len(sorted_numbers) - 1):
        gap = sorted_numbers[i] - sorted_numbers[i + 1]
        if gap > max_gap:
            max_gap = gap
            target_start = sorted_numbers[i]

    return target_start, sorted_numbers[0]

# 把 fragments 分组
def group_fragments(fragments): 
    remaining_fragments = set(range(len(fragments))) # 记录未分组的行索引
    grouped = []
    max_element = max([max(x[1]) for x in fragments])

    # 记录上一组的 [start, end] 和 [start 2, end 2]
    prev_start, prev_end = None, None
    prev_start2, prev_end2 = None, None

    while remaining_fragments:
        idx = min(remaining_fragments) # 选择未分组的最小索引行
        min_frag = min(fragments[idx][1])
        max_frag = max(fragments[idx][1])
        group = [idx]
        remaining_fragments.remove(idx)
        current_group_numbers = set(fragments[idx][1])

        # 判断当前是否是连续序列
        sorted_numbers = sorted(current_group_numbers)
        is_continuous = is_continuous_sequence(sorted_numbers)

        if not is_continuous:
            # 计算范围 [start, end]
            start, end = find_largest_diff_range(sorted_numbers) # 第一组范围
        else:
            start, end = sorted_numbers[0], sorted_numbers[-1] # 直接取连续范围

        # 继承上一组的 'end + 1'
        if prev_end is not None:
            start = prev_end + 1

        # 计算 [起始点 2, 终点 2], 仅使用不在 [start, end] 里的数
        remaining_numbers = {num for num in sorted_numbers if num < start or num > end}
        if remaining_numbers: # 只有当有剩余元素时，才计算第二个范围
            max_remaining = max(remaining_numbers)
            start2, end2 = min(remaining_numbers), (max_remaining + 1 if max_remaining + 1 < max_element else max_remaining)
        else:
            start2, end2 = None, None # 没有第二组范围

        # 继承上一组的 'start2 - 1'
        if prev_start2 is not None:
            end2 = prev_start2 - 1

        # 继续向组内添加符合条件的行
        updated = True
        while updated:
            updated = False

            for i in remaining_fragments.copy():
                frag_numbers = set(fragments[i][1])

                # 1. 若该行与组内片段有交集，则加入组
                if current_group_numbers & frag_numbers or any(
                    (start is not None and end is not None and start != end and start <= num <= end) or
                    (start2 is not None and end2 is not None and start2 != end2 and start2 <= num <= end2)
                    for num in frag_numbers                    
                ):
                    # 立即更新
                    remaining_fragments.remove(i)
                    group.append(i)
                    current_group_numbers.update(fragments[i][1])

                    sorted_numbers = sorted(current_group_numbers)
                    is_continuous = is_continuous_sequence(sorted_numbers)

                    if not is_continuous:
                        start, end = find_largest_diff_range(sorted_numbers)
                    else:
                        start, end = sorted_numbers[0], sorted_numbers[-1]

                    remaining_numbers = {num for num in sorted_numbers if num < start or num > end}
                    if remaining_numbers:
                        max_remaining = max(remaining_numbers)
                        start2, end2 = min(remaining_numbers), (max_remaining + 1 if max_remaining + 1 < max_element else max_remaining)
                    else:
                        start2, end2 = None, None

                    updated = True # 继续更新，直到所有符合条件的片段都被添加
                
                elif fragments[idx][2] == 'Non-ring' and fragments[i][3] != 'Non-bracket':
                    if start2 and (end + 1) < start2 and any(
                        (start is not None and end is not None and start != end + 1 and start <= num <= end + 1) or
                        (start2 is not None and end2 is not None and start2 != end2 and start2 <= num <= end2)
                        for num in frag_numbers                          
                    ):
                        remaining_fragments.remove(i)
                        group.append(i)

                        updated = True                        

        # 记录当前组的 [start, end] 和 [start 2, end 2] 作为下一组的参考
        prev_start, prev_end = start, end
        prev_start2, prev_end2 = start2, end2
        
        grouped.append(group)

    # 根据索引分组 fragments
    grouped_fragments = [[fragments[i] for i in group] for group in grouped]
    
    return grouped_fragments

# 进行分组的二次合并
def merge_single_groups(grouped_fragments):
    merged_groups = []
    current_group = []

    for group in grouped_fragments:
        if len(group) == 1: # 当前组只有一个元素
            current_group.extend(group) # 合并到当前 group
        else:
            if current_group: # 如果当前 group 有内容，先保存
                merged_groups.append(current_group)
                current_group = []
            merged_groups.append(group) # 直接添加多元素的 group

    if current_group: # 处理最后一个 group (如果仍有单元素组未存入)
        merged_groups.append(current_group)

    return merged_groups

class ParseState:
    def __init__(self):
        self.result = []
        self.frag_pos = [] 
        self.right_brackets = []
        self.branch_end_pos = []

        self.i = 0
        self.num = 0
        self.multi_atoms_flag = 0
        self.idx_ring_4 = 0

        self.flag = False
        self.flag_chain = False
        self.arrow_flag = False
        self.ring_bracket_flag = False

# 处理第一行为环
def handle_first_ring(state: ParseState,
                      bracket_frag,
                      current,
                      atom_indices_result):
    """
    原 first line - Ring 处理逻辑
    """    
    frag_start_end = sorted(bracket_frag[state.i][1])
    is_continuous = is_continuous_sequence(frag_start_end)

    if not is_continuous: # 计算范围 [start, end]
        start, end = find_largest_diff_range(frag_start_end) # 第一组范围
    else:
        start, end = frag_start_end[0], frag_start_end[-1] # 直接取连续范围

    if start == end:
        curr_atom_inner = None # 初始化当前原子
        for idx in bracket_frag[state.i][1]: # 找到当前原子 curr_atom_inner
            for atom_symbol, index in atom_indices_result:
                if index == idx:
                    curr_atom_inner = atom_symbol
                    break
        state.result.append('^')
        state.result.append(curr_atom_inner)
    else:
        state.result.append('^')
    state.result.append(current[0])
    handle_branch_end_after_append(state, current, state.result)    

    state.frag_pos.append(list(current[1]))
    state.flag = True
    state.num += 1
    
    if state.i + 1 < len(bracket_frag) and bracket_frag[state.i + 1][2] != 'Ring':
        if bracket_frag[state.i + 1][3] != 'Non-bracket': # 环上支链
            state.ring_bracket_flag = True
        state.i += 1
        return True
    return False

# 寻找支链结束原子位置
def bracket_maxval_flag(bracket_frag, idx):
    ring_info = bracket_frag[idx][3]
    if ring_info == 'Non-bracket':
        return max(bracket_frag[idx][1])
    pairs = ring_info[1]
    first_elements = [p[0] for p in pairs]
    if not first_elements:
        return max(bracket_frag[idx][1])
    max_val = max(first_elements)
    return max_val

# 判断函数：环引导支链
def ring_leads_branch(bracket_frag, idx):
    """
    判断 bracket_frag[idx] 这一行的 Ring 是否引导支链
    """
    atom_indices = bracket_frag[idx][1] # 第二列：原子索引集合
    ring_info = bracket_frag[idx][3] # 第四列：环连接支链信息

    if ring_info == 'Non-bracket':
        return False, None
    pairs = ring_info[1] # 获取 (x, [(a,b),(c,d)]) 特定信息
    first_elements = [p[0] for p in pairs] # 提取每个 () 中的第一个元素

    min_val = min(first_elements)
    max_val = max(first_elements)

    for idx in atom_indices: # 判断第二列是否全部落在区间内
        if idx < min_val or idx > max_val:
            return False, None 
    return True, max_val

# 检测是否到达支链结束位点并处理
def handle_branch_end_after_append(state:ParseState, 
                                   frag,
                                   target_list):
    """
    在 append fragment 之后调用，判断是否添加支链结束符 '^'
    """
    hits = [idx for idx in frag[1] if idx in state.branch_end_pos]

    if len(hits) > 1:
        raise RuntimeError(f"Multiple branch ends hit by fragment {frag}, hits={hits}")

    if hits:
        idx = hits[0]
        target_list.append('^')
        state.branch_end_pos.remove(idx)

# 作为环处理
def handle_ring_case(state: ParseState,
                     bracket_frag, 
                     current):
    """
    原 Ring 处理逻辑
    """
    if not state.flag:
        state.result.append(current[0])
        handle_branch_end_after_append(state, current, state.result)
        state.frag_pos.append(list(current[1]))
        state.num += 1
        
        if state.num != 0: # 若连接原子重复，则添加 '+'
            current_frag_pos = state.frag_pos[state.num - 1]
            if any(item in current_frag_pos for item in bracket_frag[state.i + 1][1]):
                state.result.append('+')  
                state.flag_chain = True
    ring_part = []

    if not state.flag and bracket_frag[state.i] != 'Ring':
        ring_part.append('^')
        state.arrow_flag = True
        
        ring_part.append(bracket_frag[state.i + 1][0])
        handle_branch_end_after_append(state, bracket_frag[state.i + 1], ring_part)
        state.frag_pos.append(list(bracket_frag[state.i + 1][1]))
        state.num += 1
        state.i += 1
    state.flag = False

    for j in range(state.i + 1, len(bracket_frag)): # 多环相并，3种并环方式
        if bracket_frag[j][2] == 'Ring':
            set_1 = set(bracket_frag[j - 1][1])
            set_2 = set(bracket_frag[j][1])
            common_atoms = set_1.intersection(set_2) # 计算交集原子
            common_atoms_n = len(common_atoms) # 计算交集原子个数
            
            min_ring_j= min(bracket_frag[j][1]) # 计算是否是隔环相连
            neighbor_ring_flag = False
            if (min_ring_j - 1) in bracket_frag[state.i][1]:
                neighbor_ring_flag = True
                atom_indices_result_ring = process_atoms_with_indices_after(bracket_frag[j])

            if common_atoms_n >= 4: # 连续 2 次多于 4
                state.idx_ring_4 += 1
                if state.idx_ring_4 == 2:
                    ring_part.append('+++')
                    ring_part.append(bracket_frag[j][0])
                    handle_branch_end_after_append(state, bracket_frag[j], ring_part)
                    state.frag_pos.append(list(bracket_frag[j][1]))
                    state.num += 1
                    neighbor_ring_flag = False

            if common_atoms_n == 1: # 单点连接
                ring_part.append('+')
                ring_part.append(bracket_frag[j][0])
                handle_branch_end_after_append(state, bracket_frag[j], ring_part)
            elif common_atoms_n == 2: # 共用 1 边 (2 点)
                ring_part.append('++')
                ring_part.append(bracket_frag[j][0])
                handle_branch_end_after_append(state, bracket_frag[j], ring_part)
            elif common_atoms_n == 3: # 共用 2 边 (3 点)
                ring_part.append('+++')
                ring_part.append(bracket_frag[j][0])
                handle_branch_end_after_append(state, bracket_frag[j], ring_part)
            elif neighbor_ring_flag: # 隔环相连
                element_ring = next((x[0] for x in atom_indices_result_ring if x[1] == min_ring_j), None)
                ring_part.append('^')
                ring_part.append(element_ring)
                ring_part.append(bracket_frag[j][0])
                handle_branch_end_after_append(state, bracket_frag[j], ring_part)
                state.arrow_flag = True
            elif common_atoms_n == 0: # 结束多环相并，开始下一批并环
                ring_part.append('^')
                ring_part.append(bracket_frag[j][0])
                handle_branch_end_after_append(state, bracket_frag[j], ring_part)
                state.arrow_flag = True    
            state.frag_pos.append(list(bracket_frag[j][1]))
            state.num += 1
            state.i += 1 # 跳过一行
        else:
            break
    state.result.append(''.join(ring_part))

# 作为支链处理
def handle_bracket_case(state:ParseState,
                        bracket_frag, 
                        current, 
                        atom_indices_result):
    """
    原 elif 分支逻辑
    """
    chain_head_flag = False

    if bracket_frag[state.i][3] != 'Non-bracket': # 当前行是支链
        b_f_i = bracket_frag[state.i]
        bracket_level_i = b_f_i[3][1][0][1] # 支链级别
        bracket_level_i_start = b_f_i[3][1][0][0]
        bracket_level_i_end = b_f_i[3][1][1][0]
        state.result.append('^')
    else:
        bracket_level_i = -1
        bracket_level_i_start = -1
        bracket_level_i_end = -1

    b_f = bracket_frag[state.i + 1]
    target_element = b_f[3][1]
    state.right_brackets.append(target_element[1][0])

    idx_n = 0
    for idx in current[1]: # 当前原子连接多个支链
        idx_n += 1
        if len(atom_indices_result) > 1:
            if len(atom_indices_result) % 2 == 0 and idx_n == len(atom_indices_result)/2 + 1: # 偶数个原子
                state.result.append(current[0])
                stop = state.result.copy()
                handle_branch_end_after_append(state, current, state.result)

                if isinstance(current[1], tuple):
                    state.frag_pos.append(list(current[1]))
                else:
                    state.frag_pos.append([current[1]])
                state.num += 1
                if stop != state.result:
                    break
            elif len(atom_indices_result) % 2 == 1 and idx_n == math.ceil(len(atom_indices_result)/2) + 1:
                state.result.append(current[0])
                stop = state.result.copy()
                handle_branch_end_after_append(state, current, state.result)
                if isinstance(current[1], tuple):
                    state.frag_pos.append(list(current[1]))
                else:
                    state.frag_pos.append([current[1]])
                state.num += 1
                if stop != state.result:
                    break
        curr_atom = None # 初始化当前原子

        for atom_symbol, index in atom_indices_result: # 找到当前原子 curr_atom
            if index == idx:
                curr_atom = atom_symbol
                break

        chain_parts = []
        frag_part_pos = []
        m = 1
        state.flag_chain = False
        for j in range(state.i+1, len(bracket_frag)): # 查找当前原子的下一个原子序列号 (比如 2 → 3)
            if idx + m in bracket_frag[j][1]: # 如果某行的第二列中包含 idx + 1, 则是支链
                if bracket_frag[j][3] != 'Non-bracket': # 检查该行是否是 'Bracket' 类型
                    # If the bracket's start atom != idx+m, this fragment's head
                    # atom is a backbone atom whose bracket belongs to itself, not
                    # to the current atom — stop chasing the chain here.
                    if bracket_frag[j][3][1][0][0] != idx + m:
                        break
                    b_f_inner = bracket_frag[j]
                    bracket_level_j = b_f_inner[3][1][0][1] # 支链级别
                    target_element_inner = b_f_inner[3][1]
                    state.right_brackets.append(target_element_inner[1][0]) 
                    state.i += 1
                    if m == 1:
                        if state.num != 0: # 若连接原子重复，则添加 '+'
                            current_frag_pos = state.frag_pos[state.num - 1]
                            if any(item in current_frag_pos for item in current[1]):
                                chain_parts.append('+')  
                                state.flag_chain = True  

                        if bracket_level_j != bracket_level_i - 1:
                            chain_parts.append('^')
                            chain_parts.append(curr_atom)
                            chain_parts.append(bracket_frag[j][0])
                            handle_branch_end_after_append(state, bracket_frag[j], chain_parts)
                            state.arrow_flag = True
                        elif bracket_frag[state.i - 1][3] != 'Non-bracket' and not state.ring_bracket_flag:   
                            chain_parts.append(curr_atom)
                            chain_head_flag = True
                        elif state.ring_bracket_flag:
                            chain_parts.append(curr_atom)
                            chain_parts.append(bracket_frag[j][0])
                            handle_branch_end_after_append(state, bracket_frag[j], chain_parts)

                        if state.multi_atoms_flag == 1:
                            frag_part_pos.extend([index] + list(bracket_frag[j][1]) if isinstance(bracket_frag[j][1], tuple) else [index, bracket_frag[j][1]])
                        else:
                            frag_part_pos.extend([index, bracket_frag[j][1][0]])
                        m += 1
                    elif chain_head_flag:
                        chain_parts.append('^')
                        chain_parts.append(curr_atom)
                        chain_parts.append(bracket_frag[j][0])
                        handle_branch_end_after_append(state, bracket_frag[j], chain_parts)
                        chain_head_flag = False
                        m += 1
                    else:
                        chain_parts.append('|')
                        chain_parts.append(curr_atom)
                        chain_parts.append(bracket_frag[j][0])
                        handle_branch_end_after_append(state, bracket_frag[j], chain_parts)
                        frag_part_pos.extend([index, bracket_frag[j][1][0]])
                        m += 1
                elif m == 1 and state.multi_atoms_flag != 1:
                    if state.num != 0: # 若连接原子重复，则添加 '+'
                        current_frag_pos = state.frag_pos[state.num - 1]
                        if any(item in current_frag_pos for item in current[1]):
                            state.result.append('+')
                    state.result.append(current[0])
                    handle_branch_end_after_append(state, current, state.result)
                    frag_part_pos.extend(list(current[1]))
                    break
                elif m != 1:
                    break
        
        if chain_parts: # 如果找到支链，添加到结果中
            if state.num != 0 and state.i != 1: # 若连接原子重复，则添加 '+'
                current_frag_pos = state.frag_pos[state.num - 1]
                if set(frag_part_pos) & set(current_frag_pos) and not state.flag_chain:
                    state.result.append('+')

            if state.i + 1 < len(bracket_frag): # 检查下一个未标记为支链的
                if bracket_frag[state.i + 1][3] == 'Non-bracket' and bracket_level_i_start != -1:
                    if bracket_level_i_start != bracket_level_i_end:
                        chain_parts.append(bracket_frag[state.i + 1][0])
                        handle_branch_end_after_append(state, bracket_frag[state.i + 1], chain_parts)
                        state.arrow_flag = False
                        frag_part_pos.extend([bracket_frag[state.i + 1][1][0]])
                        state.i += 1
            state.result.append(''.join(chain_parts))
            state.frag_pos.append(frag_part_pos)
            state.num += 1

            if state.multi_atoms_flag == 1:
                if state.num != 0: # 若连接原子重复，则添加 '+'
                    current_frag_pos = state.frag_pos[state.num - 1]
                    if any(item in current_frag_pos for item in current[1]) and not state.flag_chain:
                        state.result.append('+')
                state.multi_atoms_flag = 0

# 前瞻找到同一原子的所有连续 bracket，返回最后一个 bracket 的尾原子序号
def true_branch_end(bracket_frag, i):
    """Return the atom index at which the branch block truly ends.
    Scans forward from bracket_frag[i+1] to collect all consecutive brackets
    that belong to the same source atom (e.g., both F of C(F)(F)).
    """
    if i + 1 >= len(bracket_frag):
        return bracket_maxval_flag(bracket_frag, i + 1)
    next_info = bracket_frag[i + 1][3]
    if next_info == 'Non-bracket':
        return bracket_maxval_flag(bracket_frag, i + 1)

    first_start = next_info[1][0][0]  # bracket 起始原子序号 (e.g., 2 for F2)
    source_atom = first_start - 1     # 该 bracket 所属的原子 (e.g., 1 for C1)

    last_max = bracket_maxval_flag(bracket_frag, i + 1)
    m = 2  # 下一个连续 bracket 应在 source_atom + m
    for j in range(i + 2, len(bracket_frag)):
        b_info = bracket_frag[j][3]
        if b_info == 'Non-bracket':
            break
        j_start = b_info[1][0][0]
        if j_start == source_atom + m:
            last_max = bracket_maxval_flag(bracket_frag, j)
            m += 1
        else:
            break
    return last_max


# 去掉 fragment 字符串的最后一个原子 (含前缀 '-')
def strip_last_atom(frag_str):
    atoms = process_atoms_with_indices_smiles(frag_str)
    if len(atoms) <= 1:
        return frag_str
    last_pos = atoms[-1][2]
    return frag_str[:last_pos].rstrip('-')


# 处理多原子 fragment 首原子带低编号子支链的情形
# 例: C-O-C (bracket=3) + F (bracket=2) → ^CF|C-O^CF|CF^F^
def handle_multi_atom_subbranch(state, bracket_frag, current, atom_indices_result):
    # 1. 开启外层 bracket
    state.result.append('^')

    # 2. 首原子 (C2) + 子支链 (F3) + 连接符 + 部分链 (C-O)
    first_atom_idx = current[1][0]
    first_atom_sym = next(s for s, i in atom_indices_result if i == first_atom_idx)
    sub_branch = bracket_frag[state.i + 1]
    state.i += 1  # 跳过子支链 fragment

    partial_chain = strip_last_atom(current[0])
    state.result.append(first_atom_sym + sub_branch[0] + '|' + partial_chain)

    # 3. 末原子 (C5) + 其子支链 → 内层 bracket
    last_atom_idx = current[1][-1]
    last_atom_sym = next(s for s, i in atom_indices_result if i == last_atom_idx)

    inner = ['^']
    m = 1
    first_sub = True
    while state.i + 1 < len(bracket_frag):
        j = state.i + 1
        if last_atom_idx + m in bracket_frag[j][1]:
            if bracket_frag[j][3] != 'Non-bracket':
                if first_sub:
                    inner.append(last_atom_sym)
                    inner.append(bracket_frag[j][0])
                    first_sub = False
                    state.i += 1
                    m += 1
                else:
                    inner.append('|')
                    inner.append(last_atom_sym)
                    inner.append(bracket_frag[j][0])
                    inner.append('^')  # 关闭内层 bracket
                    state.i += 1
                    m += 1
                    break
            else:
                break
        else:
            break

    # 4. 尾随的 Non-bracket 原子 (F8)，关闭外层 bracket
    if state.i + 1 < len(bracket_frag):
        j = state.i + 1
        if last_atom_idx + m in bracket_frag[j][1] and bracket_frag[j][3] == 'Non-bracket':
            inner.append(bracket_frag[j][0])
            inner.append('^')  # 关闭外层 bracket
            state.i += 1

    state.result.append(''.join(inner))
    state.frag_pos.append(list(current[1]))
    state.num += 1


# 转化 smiles, 加上带支链形式
def transfer_SideChain(bracket_frag):
    state = ParseState()

    while state.i < len(bracket_frag):
        current = bracket_frag[state.i]
        atom_indices_result = process_atoms_with_indices_after(current)
        state.multi_atoms_flag = 1 if len(atom_indices_result) > 1 else 0 # 查询是否多个原子
        state.idx_ring_4 = 0
        state.flag = False

        if state.i == 0 and bracket_frag[state.i][2] == 'Ring': # 第一行为环并且 [start, end] 中 start == end
            handled = handle_first_ring(state, bracket_frag, current, atom_indices_result)
            if handled:
                continue # 函数内部已执行，继续下一轮循环
        
        if state.i + 1 < len(bracket_frag) and bracket_frag[state.i + 1][2] == 'Ring': # 如果当前行的下一行是 'Ring'
            leads_branch, max_val_bracket = ring_leads_branch(bracket_frag, state.i + 1)
            if leads_branch: # 判断 Ring 是否引导支链
                state.branch_end_pos.append(max_val_bracket)
                handle_bracket_case(state, bracket_frag, current, atom_indices_result)
            else: 
                max_val_bracket = bracket_maxval_flag(bracket_frag, state.i + 1)
                state.branch_end_pos.append(max_val_bracket)
                handle_ring_case(state, bracket_frag, current) # 正常 Ring 处理

        elif (state.multi_atoms_flag == 1  # 多原子 fragment
              and current[3] != 'Non-bracket'  # 自身有 bracket
              and state.i + 1 < len(bracket_frag)
              and bracket_frag[state.i + 1][3] != 'Non-bracket'  # 下一行也有 bracket
              and bracket_frag[state.i + 1][3][0] < current[3][0]  # 下一行 bracket 编号更小（首原子的子支链）
              and min(current[1]) + 1 in bracket_frag[state.i + 1][1]):  # 首原子直接相邻
            handle_multi_atom_subbranch(state, bracket_frag, current, atom_indices_result)

        elif state.i + 1 < len(bracket_frag) and bracket_frag[state.i + 1][3] != 'Non-bracket': # 如果当前行的下一行是 'Bracket', 检查是否连接到多个支链
            max_val_bracket = true_branch_end(bracket_frag, state.i)
            state.branch_end_pos.append(max_val_bracket)
            handle_bracket_case(state, bracket_frag, current, atom_indices_result)

        elif state.num != 0: # 若没有支链
            current_frag_pos = state.frag_pos[state.num - 1]
            if state.i != 0 and any(item in current_frag_pos for item in current[1]): # 若连接原子重复，则添加 '+'
                state.result.append('+')
                state.frag_pos.append(list(current[1]))
                state.num += 1

            if state.i != 0:
                if any(item in state.right_brackets for item in current[1]): # 检查是否需要加上右括号
                    state.result.append(current[0])
                elif bracket_frag[state.i][2] == 'Ring':
                    state.result.append('^')
                    state.result.append(current[0])
                else:
                    state.result.append(current[0])
                handle_branch_end_after_append(state, current, state.result)
                state.frag_pos.append(list(current[1]))
                state.num += 1
        else:
            state.result.append(current[0])
            handle_branch_end_after_append(state, current, state.result)
            state.frag_pos.append(list(current[1]))
            state.num += 1
        state.i += 1
    return ''.join(state.result)

# 重排序环索引
def sort_ring_with_atoms(ring, ring_atoms):
    # 将 ring 和 ring_atoms 按 ring 进行排序
    sorted_idx_atoms = sorted(zip(ring, ring_atoms), key=lambda x: x[0])

    # 拆分排序后的索引和对应的原子
    sorted_ring = tuple(idx for idx, _ in sorted_idx_atoms)
    sorted_ring_atoms = [atom for _, atom in sorted_idx_atoms]

    return sorted_ring, sorted_ring_atoms

# 碳化杂环
def find_and_replace_heterocycles(smiles, fragments):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, [], False, [], smiles

    # 获取环信息
    ring_info = mol.GetRingInfo()
    ring_systems = ring_info.AtomRings()

    has_ring = len(ring_systems) > 0 # 是否有环
    has_heterocycle = False # 是否有杂环
    hetero_pos = [] # 存储杂原子的位置
    updated_smiles = smiles # 初始化
    del_symbol_idx = [] # 存储要在 smiles 里删除的符号索引

    # 找到 fragments 第二列中的最大值
    max_value_in_fragments = max(max(frag[1]) for frag in fragments)

    # 单独处理每个环
    for ring_idx, ring in enumerate(ring_systems):
        # 记录当前环的原子
        ring_atoms = [mol.GetAtomWithIdx(idx).GetSymbol() for idx in ring]
        # 重新排序，给环上索引
        ring, ring_atoms = sort_ring_with_atoms(ring, ring_atoms) # 用 ring 的索引与 ring 建立映射
        # 创建索引映射，避免重复调用ring.index()（O(n)操作）
        ring_index_map = {atom_idx: idx for idx, atom_idx in enumerate(ring)}
        ring_hetero_pos = [] # 储存当前环中的杂原子位置

        for idx_fragments, frag in enumerate(fragments):
            result = process_atoms_with_indices(frag) # 得到原子与原始索引对应关系
            # 若是 ring，查找是第几个 ring
            frag_second_col = frag[1]
            # 查找与 frag 第二列相同的元素的索引
            for idx, ring_system in enumerate(ring_systems):
                if set(ring_system) == set(frag_second_col):
                    break

            if frag[2] == 'Ring':
                # 处理特殊情况 'C(=O)NC(=O)'
                if idx_fragments + 1 < len(fragments) and fragments[idx_fragments + 1][0] == 'C(=O)NC(=O)' and len(set(frag_second_col) & set(fragments[idx_fragments + 1][1])) >= 3:
                    continue

                if idx < ring_idx:
                    continue

            # 检查是否为全碳
            non_carbon_atoms = [(atom, idx) for atom, idx in result if atom.upper() != 'C']

            # 检查是否有元素在 ring 中
            elements_in_ring = [elem for elem in frag[1] if elem in ring]

            # 检查是否有元素不在 ring 中
            elements_not_in_ring = [elem for elem in frag[1] if elem not in ring]

            # 检查 frag 第二列的所有元素是否都在 [min_ring, max_ring] 范围内
            # 若都在范围内，且至少一个元素在该环上，则为环上支链
            # 若有不在范围内的，则为主链
            min_ring, max_ring = min(ring), max(ring)
            ring_chain_type_flag = False

            if all(min_ring <= x <= max_ring for x in frag[1]) and len(elements_in_ring) >= 1:
                ring_chain_type_flag = True

            # 处理 ring 的开头和结尾的特殊性
            # 若 min_ring 和 min_ring - 1 在同一个 frag 里，则当前 frag 也处理成环上支链
            # 若 max_ring 和 max_ring + 1 在同一个 frag 里，则当前 frag 也处理成环上支链
            if (min_ring != 0) and (min_ring in frag[1]):
                # 使用any()和生成器表达式，找到第一个匹配就停止
                if any(min_ring in frag_min_ring[1] and (min_ring - 1) in frag_min_ring[1] for frag_min_ring in fragments):
                    ring_chain_type_flag = True
            
            if (max_ring != max_value_in_fragments) and (max_ring in frag[1]):
                # 使用any()和生成器表达式，找到第一个匹配就停止
                if any(max_ring in frag_max_ring[1] and (max_ring + 1) in frag_max_ring[1] for frag_max_ring in fragments):
                    ring_chain_type_flag = True

            # frag 第二列中的每个原子都在此环上，且非全碳，则处理为环上杂原子，需要碳化
            if all(item in ring for item in frag[1]) and non_carbon_atoms:
                # 找到 non_carbon_atoms 中的元素在 ring 中的位置及其元素
                matching_elements = [
                    (ring_idx + 1, ring_index_map[atom_idx] + 1, atom, atom_idx, frag[1], 0, frag[0]) # mark 设置为 0
                    for atom, atom_idx in non_carbon_atoms
                    if atom_idx in ring_index_map
                ]
                # 处理成杂原子
                for element in matching_elements:
                    ring_hetero_pos.append(element) # 记录环的编号，杂原子位置，杂原子元素

            # frag 第二列中多个原子，只有一个在此环上，则处理为环上支链，整个 frag 碳化
            elif len(frag[1]) > 1 and len(elements_in_ring) == 1 and ring_chain_type_flag:
                # 提取此行第二个元素和后续行第二个元素，内部排序后合并在一起，
                # 若合并后连续，则不属于环上支链，而是主链
                fragments_subset = [x[1] for x in fragments[idx_fragments:]]
                fragments_subset_next = [x[1] for x in fragments[idx_fragments+1:]] if idx_fragments + 1 < len(fragments) else fragments_subset
                sorted_fragments = [tuple(sorted(item)) for item in fragments_subset]
                sorted_fragments_next = [tuple(sorted(item)) for item in fragments_subset_next]
                combined_elements = sum(sorted_fragments, ())
                combined_elements_next = sum(sorted_fragments_next, ())
                is_continuous = is_continuous_sequence(combined_elements)

                if is_continuous and not set(ring) & set(combined_elements_next): # 取消，不属于环上支链
                    continue

                # 把整个 frag 处理成杂团子
                matching_elements = [
                    (ring_idx + 1, ring_index_map[atom_idx] + 1, frag[0], atom_idx, frag[1], 1, frag[0]) # 前三个为后续用的，后三个为当前用的，mark 设置为 1
                    for atom_idx in elements_in_ring
                    if atom_idx in ring_index_map
                ]                
                # 处理成杂原子
                if matching_elements:
                    ring_hetero_pos.extend(matching_elements) # 记录环的编号，杂原子位置，杂团子
                # frag[1] 只保留 elements_in_ring 环中这一个位置

        # 如果有杂原子，记录位置并替换为碳
        if ring_hetero_pos:
            has_heterocycle = True # 标记存在杂环
            hetero_pos.extend(ring_hetero_pos)

            # 使用优化的函数提取原子表（只提取一次）
            atoms = process_atoms_with_indices_smiles(updated_smiles)
            # 只在需要修改时创建列表
            needs_update = any(pos[5] == 0 for pos in ring_hetero_pos)
            if needs_update:
                updated_smiles_list = list(updated_smiles)

            for pos in ring_hetero_pos:
                if pos[5] == 0: # 杂原子情况
                    for atom_info in atoms:
                        tag = atom_info[1] # 原子索引
                        i = atom_info[2] # 符号索引

                        if pos[3] == tag:
                            # 把原子更新为碳原子
                            updated_smiles_list[i] = 'C'
                            break

                elif pos[5] == 1: # 杂团子情况
                    # 从 pos 提取所需的值
                    value_4th = pos[3] # 第四列的值
                    min_5th = min(pos[4]) # 第五列的最小值
                    max_5th = (max(pos[4]) + 1) if max(pos[4]) < max_value_in_fragments else max(pos[4]) # 第五列的最大值加一

                    # 查找函数
                    def get_third_col(value):
                        for key, (symbol, col2, col3) in enumerate(atoms):
                            if col2 == value:
                                return col3
                        return None # 若未找到，则返回 None
                
                    # 分别获取这三个值在 atoms 里的第三列对应值
                    third_value_target = get_third_col(value_4th)
                    third_min = get_third_col(min_5th)
                    third_max = get_third_col(max_5th) 

                    # 生成 third_min 到 third_max 之间的数，但跳过 third_value_target
                    symbol = (third_max + 1) if third_max == len(updated_smiles) - 1 else third_max
                    for i in range(third_min, symbol):
                        if i != 0:
                            if i == third_max - 1 and updated_smiles[i].isdigit():
                                continue

                        if i != third_value_target:
                            del_symbol_idx.extend([i])

                    # 检查是否在括号里 
                    if third_min != 0:
                        if updated_smiles[third_min - 1] == '(':
                            del_symbol_idx.extend([third_min - 1])
                    if third_max != len(updated_smiles) - 1:
                        if updated_smiles[third_max + 1] == ')':
                            del_symbol_idx.extend([third_max + 1])
                    
                    # 去重并排序
                    del_symbol_idx = sorted(set(del_symbol_idx))
            
            # 更新 updated_smiles（如果被修改）
            if needs_update:
                updated_smiles = ''.join(updated_smiles_list)

    # 更新 updated_smiles（删除符号）
    if del_symbol_idx:
        del_symbol_idx.sort(reverse=True) # 从大到小排序，避免删除操作影响后续的索引
        updated_smiles_list_final = list(updated_smiles)
        for i in del_symbol_idx:
            updated_smiles_list_final.pop(i)
        updated_smiles = ''.join(updated_smiles_list_final)

    return has_ring, ring_systems, has_heterocycle, hetero_pos, updated_smiles

# 判断并提取环
def get_Ring(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, [] # 无效的 smiles 字符串
    
    # 计算分子中的环
    ring_info = mol.GetRingInfo()
    ring_smiles = []

    # 获取所有环的 smiles
    for ring in ring_info.AtomRings(): # 获取所有原子环
        ring_mol = Chem.PathToSubmol(mol, ring) # 提取子结构
        ring_smiles.append(Chem.MolToSmiles(ring_mol)) # 转换为 smiles

    return len(ring_smiles) > 0, list(set(ring_smiles)) # 返回布尔值，去重后的侧链，环

# 处理特殊聚合位点
def process_star(smiles):
    smiles_list = list(smiles)
    i = 0 # 当前字符的位置
    smiles_len = len(smiles_list)
    
    while i < smiles_len:
        result = process_atoms_with_indices_smiles(''.join(smiles_list[:i]))
        if i + 1 < smiles_len and ''.join(smiles_list[i:i+2]) in _ATOMIC_SYMBOLS_SET:
            if i - 1 > 0 and smiles_list[i - 1] == '(' and i + 3 < smiles_len and ''.join(smiles_list[i+2:i+4]) == '*)' and len(result) < 2: # 并且前面至少有两个原子
                smiles_list = smiles_list[:i-1] + smiles_list[i:i+2] + [smiles_list[i-1]] + smiles_list[i+2:]
                smiles_len = len(smiles_list)
                i += 4
            else:
                i += 2 # 跳过两个字符
        elif smiles_list[i].capitalize() in _ATOMIC_SYMBOLS_SET:
            if i - 1 > 0 and smiles_list[i - 1] == '(' and i + 2 < smiles_len and ''.join(smiles_list[i+1:i+3]) == '*)' and len(result) < 2: # 并且前面至少有两个原子
                smiles_list = smiles_list[:i-1] + [smiles_list[i]] + [smiles_list[i-1]] + smiles_list[i+1:]
                smiles_len = len(smiles_list)
                i += 3
            else:
                i += 1 # 跳过一个字符
        else:
            i += 1 # 如果不是原子符号，跳过该字符

    return ''.join(smiles_list)

# 显式构建括号配对关系
def build_paren_pairs(smiles: str):
    """
    构建每个 '(' 对应的 ')' 的映射
    """
    stack = []
    pairs = {}
    for i, ch in enumerate(smiles):
        if ch == "(":
            stack.append(i)
        elif ch == ")":
            if stack:
                left = stack.pop()
                pairs[left] = i
    return pairs

# 查找，保存信息，然后删除 (*)
def process_asterisk(smiles):
    # 步骤 1: 跳过第一个字符，即删掉 *
    smiles = smiles[1:]

    # 步骤 2: 找到 '(*)' 的位置
    star_pos = smiles.find('*')
    if star_pos == -1:
        return None, None, smiles

    # 步骤 3: 查找 '(*)' 前面的原子
    atoms = [] # 存储所有识别出的原子
    star_position = None # 存储 (*) 符号及其位置

    i = 0 # 当前字符的位置
    smiles_len = len(smiles)
    while i < smiles_len:
        # 尝试查找两个字符的原子符号
        if i + 1 < smiles_len and smiles[i:i+2] in _ATOMIC_SYMBOLS_SET:
            atoms.append((smiles[i:i+2], i))
            i += 2 # 跳过两个字符
        elif smiles[i].capitalize() in _ATOMIC_SYMBOLS_SET:
            atoms.append((smiles[i], i))
            i += 1 # 跳过一个字符
        elif i + 3 <= smiles_len and smiles[i:i+3] == '(*)': # 找 (*) 符号
            star_position = i # 保存 (*) 符号及其位置
            smiles = smiles[:i] + smiles[i+3:] # 步骤4: 删除 (*) 符号
            smiles_len = len(smiles)
        elif smiles[i] == '*': # 在结尾
            star_position = i
            smiles = smiles[:i] + smiles[i+1:] # 步骤4: 删除末尾的 * 符号
            smiles_len = len(smiles)
        else:
            i += 1 # 如果不是原子符号，跳过该字符
    
    # 步骤 5: 返回原子的位置和符号
    # 对每个星号 (*), 找到原子符号，并输出相应的行号和符号
    if star_position is not None:
        for ele in range(len(smiles)):
            for index, atom in enumerate(atoms):
                atom_position = atom[1] # 获取原子在 smiles 中的位置
                # 找到比 (*) 位置小 1 的原子符号
                if atom_position == star_position - 1 - ele:
                    return index + 1, atom[0], smiles # 返回原子组里的原子位置，原子符号，删掉聚合位点的 smiles
            
    return None, None, None

# 处理 (X*), 为顺序生成的预置
def process_asterisk_2(smiles: str) -> tuple[str, int]:
    # Step 1: 跳过第一个字符，即删掉 *
    smiles = smiles[1:]

    # Step 2: 找到第二个 '*' 的位置
    star_idx = smiles.find('*')
    if star_idx == -1:
        return smiles

    # Step 3: 判断 * 是否在最后
    if smiles.endswith("*"):
        # 直接进入 Step 4: 删除最后的 *
        return smiles[:-1]

    # Step 4: 以 * 为中心找最近一对()
    left = smiles.rfind("(", 0, star_idx)
    right = smiles.find(")", star_idx)
    
    if left == -1 or right == -1:
        # 没有完整 ()
        return smiles
    
    # 拿到括号块和内部内容
    bracket_block = smiles[left: right+1] # 如 "(C*)"
    inside = bracket_block[1:-1] # 如 "C*"

    # Step 5: 从 ( 左侧按"原子级规则"找最近一个原子
    j = left - 1
    atom_symbol = None
    atom_start = None

    while j >= 0:
        # 双字符优先 (Br, Cl)
        if j - 1 >= 0 and smiles[j-1:j+1] in _ATOMIC_SYMBOLS_SET:
            atom_symbol = smiles[j-1:j+1]
            atom_start = j - 1
            break
        # 单字符原子 (C, O, N, ...)
        elif smiles[j] in _ATOMIC_SYMBOLS_SET:
            atom_symbol = smiles[j]
            atom_start = j
            break
        else:
            j -= 1

    if atom_symbol is None:
        raise ValueError("❌ 在 ( 左侧未找到合法原子")
    
    # Step 6: 结构重排
    new_smiles = (
        smiles[:atom_start]         # "前一个原子" 之前的部分
        + "("                       # 把 ( 提到这个位置
        + smiles[atom_start: left]  # 原子到原来 ( 之间的内容
        + smiles[right+1:]          # 原来 ) 后面的主体部分
        + ")"                       # 补上 )
        + inside                    # 原始 inside (如 C*) 
    )

    # Step 7: 删掉第二个 *
    if new_smiles.endswith("*"):
        new_smiles = new_smiles[:-1]
    else:
        raise ValueError(f"❌ 规则违背：预期最后一个字符是 '*'，但实际为：{new_smiles}")

    # Step 8: 如果 ( 在第一个元素，逆向生成
    if new_smiles.startswith("("):
        new_smiles = (
            inside[:-1]
            + smiles[:atom_start]       # "前一个原子" 之前的部分
            + "("                       # 把 ( 提到这个位置
            + smiles[atom_start: left]  # 原子到原来 ( 之间的内容
            + smiles[right+1:]          # 原来 ) 后面的主体部分
            + ")"                       # 补上 )
        )

    return new_smiles

# 处理 (X*), 为顺序生成的预置, update
def process_asterisk_3(smiles: str) -> str:
    """
    处理形如 *X(X*)... 或 *X(*)Y... 的 SMILES
    规则：
    1. 删掉第一个 *
    2. 定位第二个 *
    3. 若是 (*)：
       - 如果被某个外层 () 包裹，且该外层 ) 不是字符串结尾 → 直接删除 (*)
    4. 否则，向右吞并连续 (XX)，按原规则重排
    """

    # ---------- Step 1: 删掉第一个 *
    if not smiles.startswith("*"):
        return smiles
    smiles = smiles[1:]

    # ---------- Step 2: 找第二个 *
    star_idx = smiles.find("*")
    if star_idx == -1:
        return smiles

    # ---------- Step 3: 如果 * 在最后，直接删掉
    if smiles.endswith("*"):
        return smiles[:-1]

    # ---------- Step 4: 找包住 * 的最近一对 ()
    left = smiles.rfind("(", 0, star_idx)
    right = smiles.find(")", star_idx)
    if left == -1 or right == -1:
        raise ValueError("❌ '*' 未被合法括号包裹")

    inner = smiles[left + 1 : right]

    # ---------- Step 5: 构建括号配对（关键）
    paren_pairs = build_paren_pairs(smiles)

    # ---------- Step 6: 特殊规则 —— (*) 不能跨越外层 ()
    if inner == "*":
        for l, r in paren_pairs.items():
            # 找到真正包住 (*) 的外层括号
            if l < left and r > right:
                # 如果这个外层 ) 不是字符串末尾
                if r != len(smiles) - 1:
                    # 直接删除 (*)
                    return smiles[:left] + smiles[right + 1 :]
                break

    # ---------- Step 7: 向右吞并连续的 (XX)
    merge_right = right
    pos = right + 1
    while pos < len(smiles) and smiles[pos] == "(":
        close = smiles.find(")", pos)
        if close == -1:
            break
        merge_right = close
        pos = close + 1

    # ---------- Step 8: 结构重排（你的原始逻辑）
    new_smiles = (
        smiles[:left]                    # '(' 之前
        + smiles[right + 1 : merge_right + 1]  # 吞掉后续 (XX)
        + "("
        + smiles[merge_right + 1 :]
        + ")"
        + inner
    )

    # ---------- Step 9: 删除最后的 *
    if new_smiles.endswith("*"):
        new_smiles = new_smiles[:-1]
    else:
        raise ValueError(
            f"❌ 规则违背：预期最后一个字符是 '*'，但实际为：{new_smiles}"
        )

    return new_smiles

# 处理饱和氮原子 [nH]
def saturate(smiles):
    return smiles.replace('[nH]', 'n')

# 查找并替换同位素和特殊共轭表示
def find_and_replace_isotope(smiles):
    change_smiles = smiles
    positions_change_atom = []
    pattern = re.compile(r'\[([23]H)\]')
    
    # Cache the atom indices result
    smiles_result = None
    i = 0
    change_smiles_len = len(change_smiles)
    
    while i < change_smiles_len:
        if i + 3 < change_smiles_len:
            match = pattern.match(change_smiles[i:i+4])
            if match:
                if smiles_result is None:
                    smiles_result = process_atoms_with_indices_smiles(change_smiles)
                
                for ele in smiles_result:
                    if ele[2] == i + 2 and ele[0] == 'H':
                        if change_smiles[i:i+4] == '[2H]':
                            positions_change_atom.append((ele[1], 'D'))
                            change_smiles = change_smiles[:i] + 'C' + change_smiles[i+4:]
                            change_smiles_len = len(change_smiles)
                            smiles_result = None  # Invalidate cache
                            i += 1  # Skip the 'C' we just added
                            break
                        elif change_smiles[i:i+4] == '[3H]':
                            positions_change_atom.append((ele[1], 'T'))
                            change_smiles = change_smiles[:i] + 'C' + change_smiles[i+4:]
                            change_smiles_len = len(change_smiles)
                            smiles_result = None  # Invalidate cache
                            i += 1  # Skip the 'C' we just added
                            break
        i += 1

    return positions_change_atom, change_smiles

def main():
    # 输入的 SMILES 字符串
    if len(sys.argv) > 1:
        # 获取传入的 smiles
        smiles = sys.argv[1]
    else:
        # 默认测试用 smiles
        # 案例
        # smiles = "*C(C*)c1ccc(cc1)COCCO"
        # smiles = "*C(C*)CCOCCO"
        # smiles = "*Cc1ccc(C*)cc1"
        # smiles = "*C(F)C(*)(F)OC(F)(F)F" # 4. 一原子多支链
        # smiles = "*CC(F)(F)C(F)(F)C(F)(F)COC(=O)CCCCC(=O)O*" # 5. 一原子多支链
        smiles = "*c1ccc(Oc2ccc(-c3ccc(Oc4ccc(N5C(=O)c6ccc(Oc7ccc(C8(c9ccc(Oc%10ccc%11c(c%10)C(=O)N(*)C%11=O)cc9)CCC(c9ccccc9)CC8)cc7)cc6C5=O)cc4)cc3C)c(C)c2)cc1"
        ################
        # smiles = "*OC(C)C(=O)*" # PLA
        # smiles = "*OCC(=O)*" # PGA
        # smiles = "*OCCCCOC(=O)CCC(=O)*" # PBS
        # smiles = "*OCCCCOC(=O)CCCCC(=O)*" # PBAT1
        # smiles = "*OCCCCOC(=O)c1ccc(cc1)C(=O)*" # PBAT2
        # smiles = "*OCCCCCC(=O)*" # PCL
        # smiles = "*OC(C)CC(=O)*" # PHB
        # smiles = "*OCCCCOC(=O)CCCCC(=O)*" # PBA
        # smiles = "*OC(C)CC(=O)*" # PHBV1
        # smiles = "*OC(CC)CC(=O)*" # PHBV2

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    img = Draw.MolToImage(mol, size=(700,500))

    origin_smiles = smiles

    # 处理 (*), 为顺序生成的预置, update
    smiles = process_asterisk_3(smiles)

    # 定义识别的官能团和原子团的 SMARTS 模式，作为官能团词表
    FunGroup = pd.read_csv(os.path.join(_GROVE_DIR, "FunGroup_token.csv"), header=None, dtype=str, sep=',')
    FunGroup = FunGroup.iloc[1:, :]

    # 处理饱和氮 [nH]
    smiles = saturate(smiles)

    # 转换同位素 [2H] [3H]
    isotope_positions, smiles = find_and_replace_isotope(smiles)

    # 提取官能团及其位置
    fragments = extract_FunGroup(smiles, FunGroup)

    # 检测官能团种类及数量，并计算溶解度
    delta = calc_delta_from_fragments(fragments)

    # 检测环并处理杂环
    has_ring, ring_systems, has_heterocycle, hetero_positions, updated_smiles = find_and_replace_heterocycles(smiles, fragments)

    # 再次提取官能团及其位置
    fragments_again = extract_FunGroup(updated_smiles, FunGroup)

    # 按照序号、交集元素和连续度，重新排序
    sorted_fragments = sort_fragments(fragments_again)

    # 调整非环类在环类中的位置，不改变非环类内原始相对位置
    sorted_fragments_ringclass = sort_fragments_ringclass(sorted_fragments)
    
    # 检查括号
    bracket_fragments = check_brackets(updated_smiles, sorted_fragments_ringclass, ring_systems)

    # 处理碳化后的杂环
    hetero_fragments = process_hetero(bracket_fragments, hetero_positions)

    # 处理碳化后的同位素、特殊表述
    isotope_hetero_fragments = process_isotope(hetero_fragments, isotope_positions)
    # 把 fragments 分组
    # grouped_fragments = group_fragments(isotope_hetero_fragments)
    # final_groups = grouped_fragments
    final_groups = isotope_hetero_fragments
        
    # 使用列表收集结果，最后join（比字符串+=更高效）
    # transfer_string_parts = []
    # for group in final_groups:
        # transfer_string_parts.append(transfer_SideChain(group))
    # transfer_string_parts.append(transfer_SideChain(final_groups))
    transfer_string_parts = [transfer_SideChain(final_groups)]

    transfer_string_with_SideChain = ''.join(transfer_string_parts)
    print(transfer_string_with_SideChain)

    img.show()
    c = 1

def smiles_to_grove(smiles: str, fungroup_csv: str = None) -> str:
    """Convert a polymer SMILES string to a GROVE token sequence.

    Args:
        smiles: Polymer SMILES with repeat-unit wildcards (*).
        fungroup_csv: Path to FunGroup_token.csv. Defaults to the bundled file.

    Returns:
        GROVE string (space-separated functional-group tokens).
    """
    if fungroup_csv is None:
        fungroup_csv = os.path.join(_GROVE_DIR, "FunGroup_token.csv")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    FunGroup = pd.read_csv(fungroup_csv, header=None, dtype=str, sep=',')
    FunGroup = FunGroup.iloc[1:, :]

    smiles = process_asterisk_3(smiles)
    smiles = saturate(smiles)
    isotope_positions, smiles = find_and_replace_isotope(smiles)
    fragments = extract_FunGroup(smiles, FunGroup)
    _ = calc_delta_from_fragments(fragments)
    has_ring, ring_systems, has_heterocycle, hetero_positions, updated_smiles = find_and_replace_heterocycles(smiles, fragments)
    fragments_again = extract_FunGroup(updated_smiles, FunGroup)
    sorted_fragments = sort_fragments(fragments_again)
    sorted_fragments_ringclass = sort_fragments_ringclass(sorted_fragments)
    bracket_fragments = check_brackets(updated_smiles, sorted_fragments_ringclass, ring_systems)
    hetero_fragments = process_hetero(bracket_fragments, hetero_positions)
    final_groups = process_isotope(hetero_fragments, isotope_positions)
    return transfer_SideChain(final_groups)


if __name__ == "__main__":
    main()



