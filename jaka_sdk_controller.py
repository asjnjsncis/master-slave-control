#!/usr/bin/env python3
"""
JAKA SDK 控制器子进程 (Python 3.8) — 极简版
============================================
使用 linear_move 进行直线运动，自带轨迹规划，无需 IK / servo_j。
"""
import os
import sys
import json
import time
import threading
import socket as sock_lib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SDK_DIR = os.path.join(SCRIPT_DIR,
                       "jakaAPI_V2.1.11_stable_20240507091715A053",
                       "SDK2.1.11", "Linux", "python3", "x86_64-linux-gnu")
sys.path.insert(0, SDK_DIR)

import jkrc


# 运动模式常量
ABS = 0
INCR = 1


class JakaSDKController:
    def __init__(self):
        self.rc = None

    def _send(self, data):
        line = json.dumps(data, separators=(",", ":"))
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    def connect(self, ip):
        # 停止可能正在运行的程序
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

        # 进入伺服模式 (servo_p 需要)
        self.rc.servo_move_enable(1)

        self._send({"type": "ready"})

    def handle_cmd(self, cmd):
        cmd_type = cmd.get("type", "")

        if cmd_type == "servo_p":
            try:
                pose = cmd["pose"]
                ret = self.rc.servo_p(pose, 0)
                if ret[0] != 0:
                    self._send({"type": "servo_p_err", "message": f"ret={ret}, 尝试恢复"})
                    try:
                        # 完整恢复流程: 清错 → 上电 → 使能伺服 → 重试
                        self.rc.clear_error()
                        self.rc.collision_recover()
                        time.sleep(0.1)
                        self.rc.power_on()
                        time.sleep(0.1)
                        self.rc.enable_robot()
                        time.sleep(0.1)
                        self.rc.servo_move_enable(1)
                        time.sleep(0.05)
                        ret2 = self.rc.servo_p(pose, 0)
                        if ret2[0] != 0:
                            self._send({"type": "servo_p_err", "message": f"恢复后仍失败 ret={ret2}, 跳过此帧"})
                        else:
                            self._send({"type": "servo_p_recovered", "message": "已恢复"})
                    except Exception as e2:
                        self._send({"type": "servo_p_err", "message": f"恢复失败: {e2}"})
            except Exception as e:
                self._send({"type": "servo_p_err", "message": str(e)})

        elif cmd_type == "get_tcp":
            try:
                _, tcp = self.rc.get_tcp_position()
                self._send({"type": "tcp", "tcp": [float(x) for x in tcp]})
            except Exception as e:
                self._send({"type": "tcp_err", "message": str(e)})

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
    print("[DEBUG] child process started", file=sys.stderr, flush=True)
    ctrl = JakaSDKController()
    init_line = sys.stdin.readline()
    if not init_line:
        print("[DEBUG] no init line received, exiting", file=sys.stderr, flush=True)
        return
    print(f"[DEBUG] got init line: {init_line.strip()}", file=sys.stderr, flush=True)
    init_data = json.loads(init_line.strip())
    ctrl.connect(init_data.get("ip", "10.5.5.100"))
    print("[DEBUG] connect completed", file=sys.stderr, flush=True)

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

    print("[DEBUG] main loop started", file=sys.stderr, flush=True)
    while True:
        cmd = _latest.pop("cmd", None)
        if cmd is None:
            time.sleep(0.0001)
            continue
        print(f"[DEBUG] got cmd: {cmd.get('type', '?')}", file=sys.stderr, flush=True)
        try:
            if not ctrl.handle_cmd(cmd):
                break
        except Exception as e:
            print(f"[DEBUG] handle_cmd error: {e}", file=sys.stderr, flush=True)
            ctrl._send({"type": "err", "message": str(e)})


if __name__ == "__main__":
    main()