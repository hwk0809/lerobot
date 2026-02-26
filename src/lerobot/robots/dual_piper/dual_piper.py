#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from functools import cached_property
from typing import Any
import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_dual_piper import DualPiperConfig


# Piper global factor 
ARM_FACTOR = 57295.779513 # 1000*180/np.pi
GRIPPER_UNIT_FACTOR = 1000.0 * 1000.0  # unit 0.001mm
GRIPPER_MAX = 0.1
GRIPPER_FACTOR = GRIPPER_UNIT_FACTOR * GRIPPER_MAX

# piper sdk
from piper_sdk import *

logger = logging.getLogger(__name__)


import sys
sys.path.insert(0, '/home/ps/workspace/whr/deformable_bench')
from common.vision_utils import process_point_cloud

class DualPiper(Robot):
    """
    Designed by cfy, jzh
    """

    config_class = DualPiperConfig
    name = "dual_piper"

    def __init__(self, config: DualPiperConfig):
        super().__init__(config)
        self.config = config
        # 创建 robot 连接
        self.piper_left = C_PiperInterface_V2("can_left")
        self.piper_right = C_PiperInterface_V2("can_right")
        # 创建 motor 映射
        self.motors = {
            "left_joint_1.pos": 0.0,
            "left_joint_2.pos": 0.0,
            "left_joint_3.pos": 0.0,
            "left_joint_4.pos": 0.0,
            "left_joint_5.pos": 0.0,
            "left_joint_6.pos": 0.0,
            "left_gripper.pos": 0.0,
            "right_joint_1.pos": 0.0,
            "right_joint_2.pos": 0.0,
            "right_joint_3.pos": 0.0,
            "right_joint_4.pos": 0.0,
            "right_joint_5.pos": 0.0,
            "right_joint_6.pos": 0.0,
            "right_gripper.pos": 0.0,
        }
        # 创建相机
        self.cameras = make_cameras_from_configs(config.cameras)
        # 创建 robot 是否连接的标志位
        self.is_robot_connected = False

        self.point_cloud_camera = None
        if hasattr(config, 'point_cloud') and config.point_cloud.enabled:
            self._init_point_cloud_camera()
        
    @property
    def _point_cloud_ft(self) -> dict[str, tuple]:
        if self.point_cloud_camera is None:
            return {}
        
        pcd_cfg = self.config.point_cloud
        return {
            "observation.point_cloud": {
                "dtype": "float32",
                "shape": (pcd_cfg.num_points, 3),
                "names": None
            }
        }
    
    @property
    def _motors_ft(self) -> dict[str, type]:
        return {k: float for k in self.motors.keys()}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft, **self._point_cloud_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        cameras_connected = all(cam.is_connected for cam in self.cameras.values())
        

        pcd_connected = (self.point_cloud_camera is None or 
                        self.point_cloud_camera.is_connected)
        return self.is_robot_connected and cameras_connected and pcd_connected

    def connect(self, calibrate: bool = True) -> None:
        """
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        """
        # 如果 robot 和 cam 都已成功连接, 则报错
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # robot connect
        self.piper_left.ConnectPort()
        self.piper_right.ConnectPort()
        self.is_robot_connected = True

        for cam in self.cameras.values():
            cam.connect()

        # ✅ 点云相机连接
        if self.point_cloud_camera is not None:
            logger.info("Connecting point cloud sensor...")
            self.point_cloud_camera.connect()
            logger.info("✅ Point cloud sensor connected")
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return True
    
    def calibrate(self):
        # 暂时空实现
        pass

    def configure(self):
        # 暂时空实现
        pass

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()
        # ==== 左臂 ====
        left_joint_state = self.piper_left.GetArmJointMsgs()
        self.motors["left_joint_1.pos"] = round(left_joint_state.joint_state.joint_1 / ARM_FACTOR, 8)
        self.motors["left_joint_2.pos"] = round(left_joint_state.joint_state.joint_2 / ARM_FACTOR, 8)
        self.motors["left_joint_3.pos"] = round(left_joint_state.joint_state.joint_3 / ARM_FACTOR, 8)
        self.motors["left_joint_4.pos"] = round(left_joint_state.joint_state.joint_4 / ARM_FACTOR, 8)
        self.motors["left_joint_5.pos"] = round(left_joint_state.joint_state.joint_5 / ARM_FACTOR, 8)
        self.motors["left_joint_6.pos"] = round(left_joint_state.joint_state.joint_6 / ARM_FACTOR, 8)

        left_gripper_raw = self.piper_left.GetArmGripperMsgs().gripper_state.grippers_angle
        self.motors["left_gripper.pos"] = round(left_gripper_raw /GRIPPER_FACTOR, 8)

        # ==== 右臂 ====
        right_joint_state = self.piper_right.GetArmJointMsgs()
        self.motors["right_joint_1.pos"] = round(right_joint_state.joint_state.joint_1 / ARM_FACTOR, 8)
        self.motors["right_joint_2.pos"] = round(right_joint_state.joint_state.joint_2 / ARM_FACTOR, 8)
        self.motors["right_joint_3.pos"] = round(right_joint_state.joint_state.joint_3 / ARM_FACTOR, 8)
        self.motors["right_joint_4.pos"] = round(right_joint_state.joint_state.joint_4 / ARM_FACTOR, 8)
        self.motors["right_joint_5.pos"] = round(right_joint_state.joint_state.joint_5 / ARM_FACTOR, 8)
        self.motors["right_joint_6.pos"] = round(right_joint_state.joint_state.joint_6 / ARM_FACTOR, 8)

        right_gripper_raw = self.piper_right.GetArmGripperMsgs().gripper_state.grippers_angle
        self.motors["right_gripper.pos"] = round(right_gripper_raw /GRIPPER_FACTOR, 8)
        
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")
        obs_dict = self.motors.copy()

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")
        
        # ✅ Capture point cloud
        if self.point_cloud_camera is not None:
            start = time.perf_counter()
            
            try:
                # 异步读取原始点云（已应用外参）
                raw_pcd = self.point_cloud_camera.async_read(timeout_ms=2000)
                
                # ✅ 使用你的处理函数
                from common.pcd_utils import process_point_cloud
                processed_pcd = process_point_cloud(
                    raw_pcd,
                    num_points=self.config.point_cloud.num_points,
                    use_gpu=False,
                    visualize=False
                )
                obs_dict["observation.point_cloud"] = processed_pcd

                dt_ms = (time.perf_counter() - start) * 1e3
                logger.info(
                f"Point cloud captured: "
                f"raw={raw_pcd.shape[0]} points, "
                f"processed={processed_pcd.shape[0]} points, "
                f"time={dt_ms:.1f}ms, "
                f"range=[X:{processed_pcd[:, 0].min():.3f}~{processed_pcd[:, 0].max():.3f}, "
                f"Y:{processed_pcd[:, 1].min():.3f}~{processed_pcd[:, 1].max():.3f}, "
                f"Z:{processed_pcd[:, 2].min():.3f}~{processed_pcd[:, 2].max():.3f}]"
                )
                
                dt_ms = (time.perf_counter() - start) * 1e3
                logger.debug(f"{self} read point_cloud: {dt_ms:.1f}ms")
                
            except Exception as e:
                logger.error(f"Failed to read point cloud: {e}")
                # 失败时返回空点云
                obs_dict["observation.point_cloud"] = np.zeros(
                    (self.config.point_cloud.num_points, 3), dtype=np.float32
                )

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.motors is None:
            raise DeviceNotConnectedError(f"self motor value is None.")
        
        action_dict = self.motors.copy()
        return action_dict
    
    def _init_point_cloud_camera(self):
        """初始化点云相机"""
        pcd_cfg = self.config.point_cloud
        
        if pcd_cfg.camera_type == "photoneo":
            from lerobot.cameras.photoneo import PhotoneoCamera, PhotoneoCameraConfig
            camera_config = PhotoneoCameraConfig(
                device_id=pcd_cfg.device_id,
                num_points=pcd_cfg.num_points,
                fps=pcd_cfg.fps,
                width=pcd_cfg.width,
                height=pcd_cfg.height,
                translation=pcd_cfg.translation,
                quaternion=pcd_cfg.quaternion,
                calibration_path=pcd_cfg.calibration_path,
            )
            self.point_cloud_camera = PhotoneoCamera(camera_config)
            logger.info(f"✅ Photoneo point cloud sensor initialized: {pcd_cfg.device_id}")
        

        elif pcd_cfg.camera_type == "zed":
            from lerobot.cameras.zed import ZedCamera
            
            self.point_cloud_camera = ZedCamera(
                serial_number=pcd_cfg.device_id,
                camera_translation=pcd_cfg.translation,
                camera_quaternion=pcd_cfg.quaternion,
                resolution=pcd_cfg.resolution,
                depth_mode=pcd_cfg.depth_mode,
            )
            logger.info(f"✅ ZED point cloud camera initialized")
        
        else:
            raise ValueError(f"Unsupported point cloud camera: {pcd_cfg.camera_type}")


    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        for cam in self.cameras.values():
            cam.disconnect()
        
        if self.point_cloud_camera is not None:
            logger.info("Disconnecting point cloud sensor...")
            self.point_cloud_camera.disconnect()
            logger.info("✅ Point cloud sensor disconnected")

        logger.info(f"{self} disconnected.")
