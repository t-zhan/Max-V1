import base64
import os

import carla
import cv2
import numpy as np
import requests

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from autopilot import AutoPilot
from transfuser_utils import PIDController

CAM_IDS = [
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT",
]


class MaxAutoPilot(AutoPilot):

    def sensors(self):
        return [
            {"type": "sensor.opendrive_map", "reading_frequency": 1e-6, "id": "hd_map"},
            {"type": "sensor.other.imu", "x": 0.0, "y": 0.0, "z": 0.0,
             "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "sensor_tick": 0.05, "id": "IMU"},
            {"type": "sensor.speedometer", "reading_frequency": 20, "id": "speed"},
            # camera rgb  -- position/rotation from DataAgent:1059-1098
            {"type": "sensor.camera.rgb", "x": 0.27, "y": -0.55, "z": 1.60,
             "roll": 0.0, "pitch": 0.0, "yaw": -55.0,
             "width": 1600, "height": 900, "fov": 70, "id": "CAM_FRONT_LEFT"},
            {"type": "sensor.camera.rgb", "x": 0.80, "y": 0.0, "z": 1.60,
             "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
             "width": 1600, "height": 900, "fov": 70, "id": "CAM_FRONT"},
            {"type": "sensor.camera.rgb", "x": 0.27, "y": 0.55, "z": 1.60,
             "roll": 0.0, "pitch": 0.0, "yaw": 55.0,
             "width": 1600, "height": 900, "fov": 70, "id": "CAM_FRONT_RIGHT"},
            {"type": "sensor.camera.rgb", "x": -0.32, "y": 0.55, "z": 1.60,
             "roll": 0.0, "pitch": 0.0, "yaw": 110.0,
             "width": 1600, "height": 900, "fov": 70, "id": "CAM_BACK_RIGHT"},
            {"type": "sensor.camera.rgb", "x": -2.0, "y": 0.0, "z": 1.60,
             "roll": 0.0, "pitch": 0.0, "yaw": 180.0,
             "width": 1600, "height": 900, "fov": 110, "id": "CAM_BACK"},
            {"type": "sensor.camera.rgb", "x": -0.32, "y": -0.55, "z": 1.60,
             "roll": 0.0, "pitch": 0.0, "yaw": -110.0,
             "width": 1600, "height": 900, "fov": 70, "id": "CAM_BACK_LEFT"},
        ]

    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        super().setup(path_to_conf_file, route_index, traffic_manager)
        self._control_config = self.config.vlm_config["BRIDGE_CONTROL"]

        turn_pid = self._control_config["turn_pid"]
        speed_pid = self._control_config["speed_pid"]
        self._turn_controller = PIDController(**turn_pid)
        self._speed_controller = PIDController(**speed_pid)
        self._max_url = os.environ["MAX_SERVER_URL"]

    def _get_control(self, input_data, vqa_data, plant):
        # ── 1. ego state ──
        tick_data = self.tick_autopilot(input_data)
        ego_speed = tick_data["speed"]
        ego_position = tick_data["gps"]

        # ── 2. read command (advance AFTER control, carla_garage order) ──
        command_idx = self.commands[-2]

        # ── 3. inference (20Hz, every tick) ──
        jpegs = []
        for cam_id in CAM_IDS:
            bgra = input_data[cam_id][1]          # (H, W, 4) BGRA
            bgr = bgra[:, :, :3]                  # drop alpha
            _, jpg = cv2.imencode(".jpg", bgr)
            jpegs.append(base64.b64encode(jpg).decode())
        resp = requests.post(
            f"{self._max_url}/max_predict",
            json={"images": jpegs, "ego_speed": ego_speed, "command_idx": command_idx},
        )
        wp_np = np.frombuffer(resp.content, dtype=np.float32).reshape(8, 2)

        # ── 4. control_pid (carla_garage plant.py:343-395) ──
        # tensor conversion removed; wp_np already numpy; ego_speed already scalar

        # speed estimation (hardcoded one_second_ct=2 for 2Hz waypoints)
        desired_speed = np.linalg.norm(wp_np[0] - wp_np[1]) * 2.0

        brake = ((desired_speed < self._control_config["brake_speed"])
                 or ((ego_speed / max(desired_speed, 1e-5))
                     > self._control_config["brake_ratio"]))

        # throttle (carla_garage L360-363)
        delta = np.clip(
            desired_speed - ego_speed,
            0.0,
            self._control_config["clip_delta"],
        )
        throttle = self._speed_controller.step(delta)
        throttle = np.clip(
            throttle,
            0.0,
            self._control_config["clip_throttle"],
        )
        throttle = 0.0 if brake else throttle

        # steering (carla_garage L365-391, tuned_aim_distance=False)
        if desired_speed < self._control_config["aim_distance_threshold"]:
            aim_distance = self._control_config["aim_distance_slow"]
        else:
            aim_distance = self._control_config["aim_distance_fast"]
        aim_index = wp_np.shape[0] - 1
        for i, wp in enumerate(wp_np):
            if np.linalg.norm(wp) >= aim_distance:
                aim_index = i
                break
        aim = wp_np[aim_index]
        angle = np.degrees(np.arctan2(aim[1], aim[0])) / 90.0
        if ego_speed < 0.01:
            angle = 0.0
        if brake:
            angle = 0.0
        steer = self._turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)

        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = throttle
        control.brake = float(brake)

        # ── 5. safety (carla_garage autopilot.py:396-408) ──
        if control.throttle == 0 and ego_speed < self.config.minimum_speed_to_prevent_rolling_back:
            control.brake = 1
        ego_velocity = CarlaDataProvider.get_velocity(self._vehicle)
        if ego_velocity < 0.1:
            self.ego_blocked_for_ticks += 1
        else:
            self.ego_blocked_for_ticks = 0
        if self.ego_blocked_for_ticks >= self.config.max_blocked_ticks:
            control.throttle = 1
            control.brake = 0

        self.steer = control.steer
        self.throttle = control.throttle
        self.brake = control.brake

        # ── 6. command advance (carla_garage autopilot.py:416-435) ──
        command_route = self._command_planner.run_step(ego_position)
        if len(command_route) > 2:
            target_point, far_command = command_route[1]
            next_target_point, next_far_command = command_route[2]
        elif len(command_route) > 1:
            target_point, far_command = command_route[1]
            next_target_point, next_far_command = command_route[1]
        else:
            target_point, far_command = command_route[0]
            next_target_point, next_far_command = command_route[0]
        if (target_point != self.target_point_prev).all():
            self.target_point_prev = target_point
            self.commands.append(far_command.value)
            self.next_commands.append(next_far_command.value)

        return control, None

    def clean_unused_folders(self):
        pass
