#!/usr/bin/env python3
"""
Alicia-D 实时坐标监控 + JAKA 目标位置计算 + CSV 录制
=====================================================
功能:
  1. 实时读取示教臂 TCP 位置 (xyz)
  2. 按相同计算链实时计算 JAKA 目标位置
  3. 同时显示 Alicia 坐标和 JAKA 目标坐标
  4. 按 s 键保存当前帧, 按 q 退出并生成 CSV

计算链 (与 alicia_jaka_xyz_only.py 一致):
  Alicia TCP → 相对零位位移 → 轴交换+符号 → 缩放 → +JAKA零位 = JAKA目标
"""

import csv
import math
import os
import signal
import sys
import time
from datetime import datetime
from typing import List, Optional

import numpy as np
from alicia_d_sdk import create_robot
from robocore.kinematics import forward_kinematics
from robocore.transform import matrix_to_euler

# ======== 配置 (与 alicia_jaka_xyz_only.py 保持一致) ========
POS_SCALE = 0.4

AXIS_MAP = [
    (1, +1.0),   # JAKA X ← Alicia Y  (正号)
    (0, -1.0),   # JAKA Y ← Alicia X  (反号)
    (2, +1.0),   # JAKA Z ← Alicia Z  (正号)
]

JAKA_ZERO_POS = [283.0, -4.0, 310.0]
JAKA_ZERO_ROT = [-1.316, 0.0, 1.571]

SOFT_LIMITS = {
    'x': (-250, 250),
    'y': (-250, 250),
    'z': (-80, 350),
}
# ===========================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 轴映射描述 (用于显示和CSV头)
AXIS_NAMES = {0: 'Alicia_X', 1: 'Alicia_Y', 2: 'Alicia_Z'}
AXIS_DESC = []
for j, (src_idx, sign) in enumerate(AXIS_MAP):
    s = '+' if sign >= 0 else '-'
    AXIS_DESC.append(f"JAKA_{['X','Y','Z'][j]} ← {s}{AXIS_NAMES[src_idx]}")


def get_alicia_tcp(leader) -> Optional[dict]:
    """读取示教臂 TCP 位姿, 返回 {x,y,z,rx,ry,rz,joints_deg}"""
    st = leader.get_robot_state("joint", cache=False)
    if st is None:
        return None
    T = forward_kinematics(leader.robot_model, st, return_end=True)
    tcp_mat = np.eye(4)
    tcp_mat[:3, :3] = T[:3, :3]
    tcp_mat[:3, 3] = T[:3, 3] * 1000
    euler = [math.degrees(a) for a in matrix_to_euler(tcp_mat[:3, :3], seq='xyz')]
    return {
        'x': tcp_mat[0, 3], 'y': tcp_mat[1, 3], 'z': tcp_mat[2, 3],
        'rx': euler[0], 'ry': euler[1], 'rz': euler[2],
        'joints_deg': [math.degrees(a) for a in st],
    }


def compute_jaka_target(alicia_pos: List[float],
                        zero_pos: List[float],
                        pos_scale: float = POS_SCALE,
                        axis_map: List[tuple] = AXIS_MAP,
                        jaka_zero_pos: Optional[List[float]] = None,
                        soft_limits: Optional[dict] = None) -> List[float]:
    """计算链: Alicia TCP 位置 → JAKA 目标位置 [x,y,z] mm"""
    if jaka_zero_pos is None:
        jaka_zero_pos = JAKA_ZERO_POS
    if soft_limits is None:
        soft_limits = SOFT_LIMITS

    # 1. 相对位移
    delta_a = [alicia_pos[i] - zero_pos[i] for i in range(3)]

    # 2. 轴交换 + 符号
    delta_swapped = [0.0, 0.0, 0.0]
    for jaka_axis, (src_idx, sign) in enumerate(axis_map):
        delta_swapped[jaka_axis] = delta_a[src_idx] * sign

    # 3. 缩放
    delta_scaled = [d * pos_scale for d in delta_swapped]

    # 4. 叠加到 JAKA 零位
    target = [jaka_zero_pos[i] + delta_scaled[i] for i in range(3)]

    # 5. 软限位
    for i, axis in enumerate(['x', 'y', 'z']):
        lo, hi = soft_limits[axis]
        rel = target[i] - jaka_zero_pos[i]
        if rel < lo:
            target[i] = jaka_zero_pos[i] + lo
        elif rel > hi:
            target[i] = jaka_zero_pos[i] + hi

    return target, delta_a, delta_swapped, delta_scaled


def save_to_csv(records: List[dict], filepath: str):
    """保存录制的帧数据到 CSV"""
    import csv
    with open(filepath, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['# Alicia-D 实时监控 + JAKA 目标计算'])
        writer.writerow(['# 录制时间', datetime.now().isoformat()])
        writer.writerow(['# 缩放系数', str(POS_SCALE)])
        for desc in AXIS_DESC:
            writer.writerow(['# 轴映射', desc])
        writer.writerow(['# JAKA 零位 (mm)'] + [f'{v:.1f}' for v in JAKA_ZERO_POS])
        writer.writerow(['# JAKA 零位姿态 (deg)'] + [f'{math.degrees(r):.1f}' for r in JAKA_ZERO_ROT])
        writer.writerow([])

        header = [
            '帧号', '时间戳_ms',
            'Alicia_X_mm', 'Alicia_Y_mm', 'Alicia_Z_mm',
            'Alicia_Rx_deg', 'Alicia_Ry_deg', 'Alicia_Rz_deg',
            'Alicia_J1', 'Alicia_J2', 'Alicia_J3', 'Alicia_J4', 'Alicia_J5', 'Alicia_J6',
            'ΔAlicia_X_mm', 'ΔAlicia_Y_mm', 'ΔAlicia_Z_mm',
            'ΔSwapped_X_mm', 'ΔSwapped_Y_mm', 'ΔSwapped_Z_mm',
            'ΔScaled_X_mm', 'ΔScaled_Y_mm', 'ΔScaled_Z_mm',
            'JAKA_X_mm', 'JAKA_Y_mm', 'JAKA_Z_mm',
        ]
        writer.writerow(header)

        for r in records:
            writer.writerow([
                r['frame'], r['timestamp_ms'],
                f'{r["alicia"][0]:.2f}', f'{r["alicia"][1]:.2f}', f'{r["alicia"][2]:.2f}',
                f'{r["alicia_rx"]:.2f}', f'{r["alicia_ry"]:.2f}', f'{r["alicia_rz"]:.2f}',
                *[f'{j:.2f}' for j in r['alicia_joints']],
                f'{r["delta_a"][0]:.2f}', f'{r["delta_a"][1]:.2f}', f'{r["delta_a"][2]:.2f}',
                f'{r["delta_swapped"][0]:.2f}', f'{r["delta_swapped"][1]:.2f}', f'{r["delta_swapped"][2]:.2f}',
                f'{r["delta_scaled"][0]:.2f}', f'{r["delta_scaled"][1]:.2f}', f'{r["delta_scaled"][2]:.2f}',
                f'{r["jaka"][0]:.2f}', f'{r["jaka"][1]:.2f}', f'{r["jaka"][2]:.2f}',
            ])


def main():
    print("=" * 70)
    print("  Alicia-D 实时监控 + JAKA 目标计算")
    print("  (实时显示 + 按键保存帧 + CSV 导出)")
    print("=" * 70)
    print()

    # ---------- 连接 ----------
    print("[1/3] 连接示教臂 Alicia-D...")
    leader = create_robot()
    if not leader.is_connected():
        print("  ❌ 连接失败!")
        return
    print("  ✅ 示教臂连接成功\n")

    # ---------- 校准零位 ----------
    print("[2/3] 校准零位...")
    print("  请将示教臂放在零位姿态, 静止不动...")
    for i in range(3, 0, -1):
        print(f"    {i}...", end=" ", flush=True)
        time.sleep(1)
    print()

    zero_pose = get_alicia_tcp(leader)
    if zero_pose is None:
        print("  ❌ 读取零位失败!")
        leader.disconnect()
        return
    zero_pos = [zero_pose['x'], zero_pose['y'], zero_pose['z']]
    print(f"\n  零位已记录: x={zero_pos[0]:.1f}  y={zero_pos[1]:.1f}  z={zero_pos[2]:.1f} mm")
    print(f"  缩放系数: {POS_SCALE}")
    for desc in AXIS_DESC:
        print(f"  轴映射: {desc}")
    print(f"  JAKA 零位: x={JAKA_ZERO_POS[0]:.1f}  y={JAKA_ZERO_POS[1]:.1f}  z={JAKA_ZERO_POS[2]:.1f} mm\n")

    # ---------- 实时监控 ----------
    print("[3/3] 开始实时监控")
    print("=" * 70)
    print("  操作:")
    print("    s — 手动标记当前帧")
    print("    q — 退出并导出全部数据到 CSV")
    print("  (运行时自动连续录制, 退出时一并保存)")
    print("=" * 70)
    print()

    records = []
    frame_count = 0
    fc, fps_s, display_s = 0, time.time(), time.time()
    running = True

    # 非阻塞键盘监听 (Linux)
    import select
    import termios
    import tty
    fd = sys.stdin.fileno()
    old_attr = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    RECORD_INTERVAL = 0.05  # 50ms ≈ 20Hz 自动录制
    last_record_time = 0

    def get_key():
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    try:
        while running:
            t_cycle = time.time()

            # 读取示教臂
            pose = get_alicia_tcp(leader)
            if pose is None:
                time.sleep(0.001)
                continue

            alicia_pos = [pose['x'], pose['y'], pose['z']]

            # 计算 JAKA 目标
            jaka_target, delta_a, delta_swapped, delta_scaled = compute_jaka_target(
                alicia_pos, zero_pos,
                pos_scale=POS_SCALE,
                axis_map=AXIS_MAP,
                jaka_zero_pos=JAKA_ZERO_POS,
                soft_limits=SOFT_LIMITS,
            )

            # === 自动录制 (每 50ms 存一帧) ===
            now = time.time()
            if now - last_record_time >= RECORD_INTERVAL:
                now_ms = int(now * 1000)
                record = {
                    'frame': frame_count,
                    'timestamp_ms': now_ms,
                    'alicia': alicia_pos,
                    'alicia_rx': pose['rx'],
                    'alicia_ry': pose['ry'],
                    'alicia_rz': pose['rz'],
                    'alicia_joints': pose['joints_deg'],
                    'delta_a': [round(v, 2) for v in delta_a],
                    'delta_swapped': [round(v, 2) for v in delta_swapped],
                    'delta_scaled': [round(v, 2) for v in delta_scaled],
                    'jaka': [round(v, 2) for v in jaka_target],
                }
                records.append(record)
                frame_count += 1
                last_record_time = now

            # 显示 (5Hz 刷新)
            fc += 1
            if time.time() - display_s >= 0.2:
                delta_a_str = f"ΔA: dx={delta_a[0]:+7.1f}  dy={delta_a[1]:+7.1f}  dz={delta_a[2]:+7.1f}"
                delta_s_str = f"ΔJ: dx={delta_scaled[0]:+7.2f}  dy={delta_scaled[1]:+7.2f}  dz={delta_scaled[2]:+7.2f}"

                print(f"\r"
                      f"🤖 Alicia: x={alicia_pos[0]:8.1f}  y={alicia_pos[1]:8.1f}  z={alicia_pos[2]:8.1f} mm  |  "
                      f"🎯 JAKA:   x={jaka_target[0]:8.1f}  y={jaka_target[1]:8.1f}  z={jaka_target[2]:8.1f} mm  |  "
                      f"{delta_a_str}  {delta_s_str}  |  已录:{len(records):>4}帧",
                      end="", flush=True)
                display_s = time.time()

            # FPS
            if time.time() - fps_s >= 5.0:
                fps = fc / (time.time() - fps_s)
                print(f"\n[INFO] 读取频率: {fps:.0f} Hz | 录制帧数: {len(records)}")
                fc, fps_s = 0, time.time()

            # 按键处理
            key = get_key()
            if key == 's':
                print(f"\n📌 已标记帧 #{frame_count - 1}  (共录 {len(records)} 帧)")
            elif key == 'q':
                print("\n\n[INFO] 用户请求退出")
                running = False

            time.sleep(max(0, 1.0 / 200.0 - (time.time() - t_cycle)))

    except KeyboardInterrupt:
        print("\n\n[INFO] Ctrl+C 中断")
    finally:
        # 恢复终端
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attr)

        # 断开
        try:
            leader.disconnect()
        except Exception:
            pass
        print("[INFO] 示教臂已断开 ✅")

        # 保存 CSV (只要有记录就保存)
        if records:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = os.path.join(SCRIPT_DIR, f"alicia_live_jaka_{timestamp}.csv")
            save_to_csv(records, csv_path)
            print(f"✅ 共保存 {len(records)} 帧 → {csv_path}")
        else:
            print("ℹ️  未保存任何帧")


if __name__ == "__main__":
    main()
