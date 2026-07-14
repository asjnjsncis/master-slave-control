#!/usr/bin/env python3
"""
JAKA SDK 控制器子进程 (Python 3.8) - 高性能伺服版
✅ 核心修复: servo_j 替代 joint_move，消除轨迹规划延迟
✅ 新增: servo_j 直接透传命令 (跳过 IK, 由主进程解算)
✅ 保留: servo_j_from_tcp (IK + servo_j, 作为备选)
✅ 安全: 软限位 + 解族跳变检测 + 单帧限幅
✅ 频率: 移除 50ms 限速, servo_j 可高达 200Hz
"""
import sys
import json
import math
import time
import os
import threading
import socket as sock_lib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SDK_DIR = os.path.join(SCRIPT_DIR,
                       "jakaAPI_V2.1.11_stable_20240507091715A053",
                       "SDK2.1.11", "Linux", "python3", "x86_64-linux-gnu")
sys.path.insert(0, SDK_DIR)

import jkrc


class JakaSDKController:
    def __init__(self):
        self.rc = None
        self.ready = False
        self._smooth_joints = []
        self._jsmooth = 0.5
        self._frame_count = 0

        # JAKA Zu3 关节软限位 (弧度) - 比机械限位更保守，匹配JAKA内部软限位
        self._joint_limits = [
            (-6.283, 6.283),    # J1: ±360° (全范围)
            (-2.094, 2.094),    # J2: ±120° (比机械限位135°更保守)
            (-2.094, 2.094),    # J3: ±120° (比机械限位135°更保守)
            (-6.283, 6.283),    # J4: ±360°
            (-1.745, 1.745),    # J5: ±100° (比机械限位120°更保守)
            (-6.283, 6.283),    # J6: ±360°
        ]
        # 每步最大关节增量 (rad) - 用于servo_j限速，约0.57°/步
        self._max_step = 0.01
        # 伺服发送间隔 (秒) - 用于计算速度限幅
        self._servo_dt = 0.01  # 假设 ~100Hz

    def _send(self, data):
        line = json.dumps(data, separators=(",", ":"))
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    def connect(self, ip):
        """连接 JAKA 并初始化"""
        try:
            s = sock_lib.socket(sock_lib.AF_INET, sock_lib.SOCK_STREAM)
            s.settimeout(1)
            s.connect((ip, 10001))
            s.sendall(b'{"cmdName":"stop_program"}\n')
            s.close()
        except Exception:
            pass

        self.rc = jkrc.RC(ip)
        self.rc.login()
        self.rc.power_on()
        self.rc.enable_robot()

        _, jpos = self.rc.get_joint_position()
        _, tcp = self.rc.get_tcp_position()
        self._send({"type": "init", "joints": [float(x) for x in jpos],
                    "tcp": [float(x) for x in tcp]})

        self.rc.set_collision_level(5)
        self.rc.clear_error()
        try:
            self.rc.collision_recover()
        except Exception:
            pass

        # 启用伺服模式 + NLF轨迹滤波 (限制速度/加速度/加加速度)
        self.rc.servo_move_enable(1)
        try:
            # NLF: 内置轨迹规划, 防止 servo_j 命令增量过大
            # max_vr=0.5 rad/s (~28°/s), max_ar=1.0 rad/s², max_jr=2.0 rad/s³
            self.rc.servo_move_use_joint_NLF(0.5, 1.0, 2.0)
        except Exception:
            pass  # 旧版SDK可能不支持
        self._send({"type": "servo_enabled", "result": "ok"})

        self._smooth_joints = [float(x) for x in jpos]
        self.ready = True
        self._send({"type": "ready"})

    def handle_cmd(self, cmd):
        cmd_type = cmd.get("type", "")

        # ========== servo_j 直接透传 (带限幅限速) ==========
        if cmd_type == "servo_j":
            joints = cmd.get("joints")
            if joints and len(joints) == 6:
                try:
                    ref = list(self._smooth_joints)
                    # 软限位
                    clamped = [max(self._joint_limits[i][0],
                                  min(self._joint_limits[i][1], joints[i]))
                               for i in range(6)]
                    # 每步限幅 (防止跳变)
                    step_limited = [
                        ref[i] + max(-self._max_step, min(self._max_step, clamped[i] - ref[i]))
                        for i in range(6)
                    ]
                    # 指数平滑
                    smooth = [self._smooth_joints[i] + self._jsmooth * (step_limited[i] - self._smooth_joints[i])
                             for i in range(6)]
                    self._smooth_joints = list(smooth)
                    self.rc.servo_j(smooth, 1)  # 1=绝对模式
                except Exception as e:
                    self._send({"type": "servo_j_err", "message": str(e)})

        # ========== servo_j_from_tcp: IK + servo_j (带多层安全限制) ==========
        elif cmd_type == "servo_j_from_tcp":
            tcp = cmd.get("tcp")
            # 主进程传过来的 max_joint_delta 用作"粗限幅"，子进程用 _max_step 做"精细限幅"
            max_joint_delta = cmd.get("max_joint_delta", 0.08)

            try:
                ref = list(self._smooth_joints)

                # 快速 IK
                ik_result = self.rc.kine_inverse(ref, tcp)

                if len(ik_result) < 2 or ik_result[0] != 0:
                    return True  # IK 失败, 保持原位

                joints = list(ik_result[1])

                # 解族跳变检测 (>5° 丢弃)
                if max(abs(joints[i] - ref[i]) for i in range(6)) > 0.087:
                    return True

                # ① 关节软限位
                clamped = [max(self._joint_limits[i][0],
                              min(self._joint_limits[i][1], joints[i]))
                          for i in range(6)]

                # ② 主进程粗限幅 (与 ref 比, 防止 IK 跳变)
                d1 = max(abs(clamped[i] - ref[i]) for i in range(6))
                if d1 > max_joint_delta:
                    s = max_joint_delta / d1
                    clamped = [ref[i] + (clamped[i] - ref[i]) * s for i in range(6)]

                # ③ 子进程精细限幅 (与上一帧平滑值比, 保证 servo_j 增量 < 0.57°/步)
                d2 = max(abs(clamped[i] - self._smooth_joints[i]) for i in range(6))
                if d2 > self._max_step:
                    s = self._max_step / d2
                    clamped = [self._smooth_joints[i] + (clamped[i] - self._smooth_joints[i]) * s
                              for i in range(6)]

                # ④ 指数平滑
                smooth = [self._smooth_joints[i] + self._jsmooth * (clamped[i] - self._smooth_joints[i])
                         for i in range(6)]

                # ⑤ 再次精细钳位 (平滑后可能略超)
                d3 = max(abs(smooth[i] - self._smooth_joints[i]) for i in range(6))
                if d3 > self._max_step:
                    s = self._max_step / d3
                    smooth = [self._smooth_joints[i] + (smooth[i] - self._smooth_joints[i]) * s
                             for i in range(6)]

                self._smooth_joints = list(smooth)
                self.rc.servo_j(smooth, 1)  # 1=绝对模式

                # 保活 (每 500 帧)
                self._frame_count += 1
                if self._frame_count % 500 == 0:
                    try:
                        self.rc.servo_move_enable(1)
                    except Exception:
                        pass

            except Exception as e:
                self._send({"type": "servo_j_err", "message": str(e)})

        elif cmd_type == "clear_error":
            try:
                self.rc.clear_error()
                self.rc.collision_recover()
                self._send({"type": "clear_ok"})
            except Exception as e:
                self._send({"type": "clear_err", "message": str(e)})

        elif cmd_type == "shutdown":
            try:
                self.rc.servo_move_enable(0)
                self.rc.logout()
            except Exception:
                pass
            self._send({"type": "shutdown"})
            return False

        return True


def main():
    ctrl = JakaSDKController()
    init_line = sys.stdin.readline()
    if not init_line:
        return
    init_data = json.loads(init_line.strip())
    ctrl.connect(init_data.get("ip", "10.5.5.100"))

    # 后台线程: 持续读取 stdin, 仅保留最新一条命令
    _latest = {"cmd": None}

    def _reader():
        for line in sys.stdin:
            if not line:
                break
            try:
                _latest["cmd"] = json.loads(line.strip())
            except json.JSONDecodeError:
                pass

    t = threading.Thread(target=_reader, daemon=True)
    t.start()

    # 主线程: 处理最新命令, 空闲时短暂休眠
    while True:
        # 原子取出并清除, 避免 reader 线程同时写入导致覆盖
        cmd = _latest.pop("cmd", None)
        if cmd is None:
            time.sleep(0.001)
            continue

        try:
            if not ctrl.handle_cmd(cmd):
                break
        except Exception as e:
            ctrl._send({"type": "servo_j_err", "message": str(e)})


if __name__ == "__main__":
    main()