#!/usr/bin/env python3
"""
Alicia-D 实时坐标监控 + JAKA servo_p 实时控制 (不经过 CSV)
============================================================
功能:
  1. 实时读取示教臂 TCP 位置 (xyz)
  2. 按计算链实时计算 JAKA 目标位置
  3. 通过 servo_p 实时发送到从臂 (不经过 CSV 文件)
  4. Enter 键切换 连接/断开 状态
  5. 仅 JAKA X 轴运动 (Y、Z 固定在零位)

计算链 (与 alicia_jaka_xyz_only.py 一致):
  Alicia TCP → 相对零位位移 → 轴交换+符号 → 缩放 → +JAKA零位 = JAKA目标

控制逻辑:
  - 默认: 主臂控制从臂 (连接状态)
  - 按 Enter: 切换断开, 主臂不再控制从臂 (从臂停在当前位置)
  - 再按 Enter: 重新连接, 主臂继续控制从臂
"""

import math
import os
import signal
import sys
import time
from typing import List, Optional

import numpy as np
from alicia_d_sdk import create_robot
from robocore.kinematics import forward_kinematics
from robocore.transform import matrix_to_euler

# ======== 配置 (与 alicia_jaka_xyz_only.py 保持一致) ========
POS_SCALE = 0.6

AXIS_MAP = [
    (0, +1.0),   # JAKA X ← Alicia X  (正号)
    (1, +1.0),   # JAKA Y ← Alicia Y  (正号)
    (2, +1.0),   # JAKA Z ← Alicia Z  (正号)
]

JAKA_ZERO_POS = [283.0, -4.0, 310.0]   # JAKA 零位 TCP 位置 (mm)
JAKA_ZERO_ROT = [-1.316, 0.0, 1.571]   # JAKA 零位 TCP 姿态 (rad) — 固定不变

SOFT_LIMITS = {
    'x': (-250, 250),
    'y': (-250, 250),
    'z': (-80, 350),
}

# 工作范围 — 超出此范围自动断开 (相对 JAKA 起始位置, mm)
WORK_RANGE = {
    'x': (-200, 200),
    'y': (-200, 200),
    'z': (-200, 200),
}

CONTROL_RATE_HZ = 50       # 控制频率
SERVO_DT = 0.008           # 每次 servo_p 发送间隔 (秒)
RECORD_INTERVAL = 0.02     # CSV 记录间隔 (秒), 50Hz
# ===========================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

AXIS_NAMES = {0: 'Alicia_X', 1: 'Alicia_Y', 2: 'Alicia_Z'}
AXIS_DESC = []
for j, (src_idx, sign) in enumerate(AXIS_MAP):
    s = '+' if sign >= 0 else '-'
    AXIS_DESC.append(f"JAKA_{['X','Y','Z'][j]} ← {s}{AXIS_NAMES[src_idx]}")


# ===========================
#  JAKA SDK 控制器 (子进程)
# ===========================

class JakaSDKController:
    """JAKA SDK 控制器 (子进程通信) — 复用 v6 中的实现"""

    def __init__(self, ip="10.5.5.100"):
        import json
        import subprocess
        import select as _select

        self._smooth_pose = None
        self._max_step = 1.5          # 单步最大变化 (mm/rad)
        self._smooth_factor = 0.4     # 指数平滑系数

        ctrl_script = os.path.join(SCRIPT_DIR, "jaka_sdk_controller.py")
        log_dir = os.path.join(SCRIPT_DIR, "logs")
        os.makedirs(log_dir, exist_ok=True)

        sdk_dir = os.path.join(
            SCRIPT_DIR,
            "jakaAPI_V2.1.11_stable_20240507091715A053",
            "SDK2.1.11", "Linux", "python3", "x86_64-linux-gnu"
        )

        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = sdk_dir + ":" + env.get("LD_LIBRARY_PATH", "")

        self.proc = subprocess.Popen(
            ["/usr/bin/python3.8", ctrl_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=open(os.path.join(log_dir, "sdk_err.log"), "w"),
            env=env,
            text=True,
            bufsize=1
        )

        self._send = lambda data: (
            self.proc.stdin.write(json.dumps(data, separators=(",", ":")) + "\n"),
            self.proc.stdin.flush()
        )

        self._send({"ip": ip})
        self._wait_ready()

    def _read(self):
        import json
        line = self.proc.stdout.readline()
        return json.loads(line.strip()) if line else None

    def _wait_ready(self):
        self.init_joints = None
        self.init_tcp = None
        while True:
            resp = self._read()
            if resp is None:
                raise RuntimeError("SDK controller died")
            t = resp.get("type", "")
            if t == "init":
                self.init_joints = resp["joints"]
                self.init_tcp = resp["tcp"]
            elif t == "servo_enabled":
                print("[JAKA] 伺服已启用 ✅")
            elif t == "ready":
                print(f"[JAKA] 初始关节: {[f'{math.degrees(a):.2f}°' for a in self.init_joints]}")
                print(f"[JAKA] 初始 TCP:  x={self.init_tcp[0]:.1f} y={self.init_tcp[1]:.1f} z={self.init_tcp[2]:.1f} mm")
                print("[JAKA] SDK 就绪 ✅")
                break
            elif t == "error":
                raise RuntimeError(f"SDK error: {resp.get('message')}")

    def servo_p(self, pose):
        """笛卡尔空间伺服运动 (绝对模式) — 带平滑与单步限幅
        pose: [x, y, z, rx, ry, rz] (mm, rad)
        """
        if self._smooth_pose is None:
            self._smooth_pose = list(pose)

        # 1. 单步限幅
        clamped = list(self._smooth_pose)
        for i in range(6):
            delta = pose[i] - self._smooth_pose[i]
            if abs(delta) > self._max_step:
                clamped[i] = self._smooth_pose[i] + math.copysign(self._max_step, delta)
            else:
                clamped[i] = pose[i]

        # 2. 指数平滑
        smoothed = [
            self._smooth_pose[i] + self._smooth_factor * (clamped[i] - self._smooth_pose[i])
            for i in range(6)
        ]

        self._smooth_pose = list(smoothed)
        self._send({"type": "servo_p", "pose": smoothed})

    def get_tcp(self) -> Optional[List[float]]:
        """读取 JAKA 当前 TCP [x,y,z, rx,ry,rz] (mm, rad)"""
        import json
        self._send({"type": "get_tcp"})
        resp = self._read()
        if resp and resp.get("type") == "tcp":
            return resp["tcp"]
        return None

    def poll_errors(self):
        """读取子进程消息 (非阻塞)"""
        import json
        import select
        had_error = False
        try:
            for _ in range(10):
                r, _, _ = select.select([self.proc.stdout], [], [], 0)
                if not r:
                    break
                line = self.proc.stdout.readline()
                if line:
                    try:
                        resp = json.loads(line.strip())
                        t = resp.get("type", "")
                        if t == "servo_p_err":
                            print(f"\n❌ servo_p 错误: {resp.get('message', '')}")
                            had_error = True
                        elif t == "servo_p_recovered":
                            print(f"\n↻ servo_p 已自动恢复 ✅")
                        elif t == "servo_p_warn":
                            print(f"\n⚠️ servo_p 队列: {resp.get('message', '')}")
                    except Exception:
                        pass
        except Exception:
            pass
        return had_error

    def stop(self):
        try:
            self._send({"type": "shutdown"})
            self.proc.wait(timeout=3)
        except Exception:
            self.proc.kill()


# ===========================
#  Alicia 示教臂读取
# ===========================

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


def compute_jaka_target(
    alicia_pos: List[float],
    zero_pos: List[float],
    jaka_start_pos: List[float],
    pos_scale: float = POS_SCALE,
    axis_map: List[tuple] = AXIS_MAP,
    jaka_zero_rot: Optional[List[float]] = None,
) -> List[float]:
    """计算链: Alicia TCP 位置 → JAKA 目标位置 [x,y,z, rx,ry,rz]

    基准: 主臂在零位时, JAKA 目标 = jaka_start_pos (实际初始位置)
          主臂移动时, 各轴按 axis_map 映射+缩放跟随,
          姿态用 jaka_zero_rot 固定
    """
    if jaka_zero_rot is None:
        jaka_zero_rot = JAKA_ZERO_ROT

    # 1. 相对位移 (Alicia 坐标系)
    delta_a = [alicia_pos[i] - zero_pos[i] for i in range(3)]

    # 2. 轴交换 + 符号 + 缩放 (全部三轴)
    target_pos = list(jaka_start_pos)
    for jaka_axis, (src_idx, sign) in enumerate(axis_map):
        delta_jaka = delta_a[src_idx] * sign * pos_scale
        target_pos[jaka_axis] += delta_jaka

    # 3. 软限位 (各轴相对起始位置)
    for i, axis in enumerate(['x', 'y', 'z']):
        lo, hi = SOFT_LIMITS[axis]
        rel = target_pos[i] - jaka_start_pos[i]
        if rel < lo:
            target_pos[i] = jaka_start_pos[i] + lo
        elif rel > hi:
            target_pos[i] = jaka_start_pos[i] + hi

    # 完整位姿: [x, y, z, rx, ry, rz]
    return target_pos + list(jaka_zero_rot)


# ===========================
#  CSV 保存
# ===========================

def save_to_csv(records: List[dict], filepath: str, jaka_start_pos: List[float]):
    """保存录制的帧数据到 CSV"""
    import csv
    from datetime import datetime
    with open(filepath, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['# Alicia-D 实时 servo_p 控制记录'])
        writer.writerow(['# 录制时间', datetime.now().isoformat()])
        writer.writerow(['# 缩放系数', str(POS_SCALE)])
        for desc in AXIS_DESC:
            writer.writerow(['# 轴映射', desc])
        writer.writerow(['# JAKA 起始位置 (mm)'] + [f'{v:.1f}' for v in jaka_start_pos])
        writer.writerow([])

        header = [
            '帧号', '时间戳_ms',
            'Alicia_X_mm', 'Alicia_Y_mm', 'Alicia_Z_mm',
            'Alicia_Rx_deg', 'Alicia_Ry_deg', 'Alicia_Rz_deg',
            'Alicia_J1', 'Alicia_J2', 'Alicia_J3', 'Alicia_J4', 'Alicia_J5', 'Alicia_J6',
            'ΔAlicia_X_mm', 'ΔAlicia_Y_mm', 'ΔAlicia_Z_mm',
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
                f'{r["delta_scaled"][0]:.2f}', f'{r["delta_scaled"][1]:.2f}', f'{r["delta_scaled"][2]:.2f}',
                f'{r["jaka"][0]:.2f}', f'{r["jaka"][1]:.2f}', f'{r["jaka"][2]:.2f}',
            ])


# ===========================
#  主程序
# ===========================

def main():
    print("=" * 70)
    print("  Alicia-D 实时监控 → JAKA servo_p 实时控制")
    print("  (不经过 CSV, 实时映射 + servo_p 控制)")
    print("=" * 70)
    print()

    # ---------- 连接示教臂 ----------
    print("[1/4] 连接示教臂 Alicia-D...")
    leader = create_robot()
    if not leader.is_connected():
        print("  ❌ 连接失败!")
        return
    print("  ✅ 示教臂连接成功\n")

    # ---------- 校准零位 ----------
    print("[2/4] 校准零位...")
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
    print(f"  轴映射 (仅 JAKA X 跟随): {AXIS_DESC[0]}")
    print(f"  JAKA 零位: x={JAKA_ZERO_POS[0]:.1f}  y={JAKA_ZERO_POS[1]:.1f}  z={JAKA_ZERO_POS[2]:.1f} mm")
    print(f"  JAKA 姿态: rx={JAKA_ZERO_ROT[0]:.3f}  ry={JAKA_ZERO_ROT[1]:.3f}  rz={JAKA_ZERO_ROT[2]:.3f} rad\n")

    # ---------- 连接 JAKA ----------
    print("[3/4] 连接从臂 JAKA...")
    follower = JakaSDKController(ip="10.5.5.100")
    print()

    # ---------- 读取初始 TCP ----------
    print("[4/4] 读取实际 TCP 作为起始位置...")
    actual_tcp = follower.init_tcp
    if actual_tcp is None:
        print("❌ 读取 TCP 失败!")
        follower.stop()
        leader.disconnect()
        return

    # 记录 JAKA 实际初始位置作为偏移基准
    init_pos = list(actual_tcp[:3])   # [init_x, init_y, init_z] 实际 TCP 位置
    init_rot = list(actual_tcp[3:])   # 实际 TCP 姿态, 避免启动时剧烈运动
    print(f"\n  初始 TCP:  x={init_pos[0]:.2f}  y={init_pos[1]:.2f}  z={init_pos[2]:.2f} mm")
    print(f"  初始姿态: rx={init_rot[0]:.3f}  ry={init_rot[1]:.3f}  rz={init_rot[2]:.3f} rad")
    print(f"  轴映射: {', '.join(AXIS_DESC)}")
    print(f"  缩放系数: {POS_SCALE}")
    print(f"  控制频率: {CONTROL_RATE_HZ} Hz")
    print()
    print("=" * 70)
    print("  状态: 🔗 已连接 (主臂控制从臂)")
    print("  操作:")
    print("    Enter — 切换 连接/断开")
    print("    q     — 退出程序")
    print("=" * 70)
    print()

    # ---------- 键盘监听设置 (Linux 非阻塞) ----------
    import select
    import termios
    import tty
    fd = sys.stdin.fileno()
    old_attr = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    # ---------- 控制状态 ----------
    connected = True          # 默认连接状态
    running = True
    follower._smooth_pose = None  # 初始化平滑位姿

    def get_key():
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    # SIGINT 处理
    def _handle_sigint(s, f):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, _handle_sigint)

    # 首次发送: 将 JAKA 固定到实际初始位置
    first_target = init_pos + init_rot
    for _ in range(5):
        follower.servo_p(first_target)
        time.sleep(SERVO_DT)

    # ---------- 主循环 ----------
    frame_count = 0
    fps_start = time.time()
    last_record_time = 0
    records = []
    servo_err_count = 0            # 连续 servo_p 错误计数
    MAX_SERVO_ERR = 10            # 超过此值重置平滑并跳过

    try:
        while running:
            t_cycle = time.time()

            # 读取示教臂
            pose = get_alicia_tcp(leader)
            if pose is None:
                time.sleep(0.001)
                continue

            alicia_pos = [pose['x'], pose['y'], pose['z']]

            # 计算 JAKA 目标 (全部三轴跟随)
            tcp_target = compute_jaka_target(alicia_pos, zero_pos, init_pos, jaka_zero_rot=init_rot)

            # 检查是否超出工作范围
            for i, axis in enumerate(['x', 'y', 'z']):
                delta = tcp_target[i] - init_pos[i]
                lo, hi = WORK_RANGE[axis]
                if delta < lo or delta > hi:
                    if connected:
                        connected = False
                        print(f"\n\n⚠️ 超出工作范围! {axis.upper()} 偏移 {delta:+.0f}mm 超出 [{lo}, {hi}]mm, 已自动断开")
                        print(f"   按 Enter 重新连接后继续控制")
                    break

            # 如果处于连接状态, 发送 servo_p
            if connected:
                follower.servo_p(tcp_target)
                if follower.poll_errors():
                    servo_err_count += 1
                    if servo_err_count >= MAX_SERVO_ERR:
                        print(f"\n\n⚠️ servo_p 连续 {MAX_SERVO_ERR} 次错误, 重置平滑并跳过此帧")
                        follower._smooth_pose = None
                        servo_err_count = 0
                else:
                    servo_err_count = 0

            # --- CSV 记录 (RECORD_INTERVAL 间隔) ---
            now = time.time()
            if now - last_record_time >= RECORD_INTERVAL:
                now_ms = int(now * 1000)
                delta_a = [alicia_pos[i] - zero_pos[i] for i in range(3)]
                delta_scaled = [tcp_target[i] - init_pos[i] for i in range(3)]
                records.append({
                    'frame': frame_count,
                    'timestamp_ms': now_ms,
                    'alicia': alicia_pos,
                    'alicia_rx': pose.get('rx', 0), 'alicia_ry': pose.get('ry', 0), 'alicia_rz': pose.get('rz', 0),
                    'alicia_joints': pose.get('joints_deg', [0]*6),
                    'delta_a': [round(d, 2) for d in delta_a],
                    'delta_scaled': [round(d, 2) for d in delta_scaled],
                    'jaka': [round(v, 2) for v in tcp_target[:3]],
                    'jaka_rot': [round(v, 4) for v in tcp_target[3:]],
                })
                last_record_time = now

            # --- 显示 (20Hz) ---
            frame_count += 1
            if time.time() - fps_start >= 0.05:
                # 读取 JAKA 实时 TCP
                real_tcp = follower.get_tcp()
                if real_tcp is None:
                    real_tcp_str = "N/A"
                else:
                    real_tcp_str = f"x={real_tcp[0]:8.1f}  y={real_tcp[1]:8.1f}  z={real_tcp[2]:8.1f}"

                status_icon = "🔗" if connected else "⛓️‍💥"
                status_text = "已连接" if connected else "已断开"
                d = [tcp_target[i] - init_pos[i] for i in range(3)]
                print(f"\r"
                      f"{status_icon} {status_text:6s}  |  "
                      f"🤖 Alicia: x={alicia_pos[0]:7.1f} y={alicia_pos[1]:7.1f} z={alicia_pos[2]:7.1f}  |  "
                      f"🎯 Jaka:  x={tcp_target[0]:7.1f} y={tcp_target[1]:7.1f} z={tcp_target[2]:7.1f}  |  "
                      f"📡 实时: {real_tcp_str}  |  "
                      f"Δ=({d[0]:+.1f},{d[1]:+.1f},{d[2]:+.1f})  |  "
                      f"帧:{frame_count:>5}",
                      end="", flush=True)
                fps_start = time.time()

            # --- 按键处理 ---
            key = get_key()
            if key == '\n' or key == '\r':  # Enter 键
                connected = not connected
                if connected:
                    print(f"\n\n🔗 已连接 — 主臂恢复控制从臂")
                    # 不重置平滑位姿, 让 servo_p 的限幅机制自然过渡
                else:
                    print(f"\n\n⛓️‍💥 已断开 — 主臂不再控制从臂 (从臂停在当前位置)")
            elif key == 'q':
                print("\n\n[INFO] 用户请求退出")
                running = False

            # 以 50Hz 频率运行
            elapsed = time.time() - t_cycle
            sleep_time = max(0, 1.0 / CONTROL_RATE_HZ - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\n[INFO] Ctrl+C 中断")
    finally:
        # 恢复终端
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attr)

        # 停止 servo
        print("\n[INFO] 停止 servo 控制...")
        try:
            follower.stop()
        except Exception:
            pass

        # 断开示教臂
        try:
            leader.disconnect()
        except Exception:
            pass

        # 保存 CSV
        if records:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = os.path.join(SCRIPT_DIR, f"alicia_servop_{timestamp}.csv")
            save_to_csv(records, csv_path, init_pos)
            print(f"\n✅ 共保存 {len(records)} 帧 → {os.path.basename(csv_path)}")

        print("[INFO] ✅ 程序已退出")


if __name__ == "__main__":
    main()
