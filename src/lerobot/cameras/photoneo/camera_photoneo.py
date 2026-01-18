from loguru import logger
import numpy as np
import time
import os
from threading import Thread, Event, Lock
from typing import Any
from numpy.typing import NDArray

from ..camera import Camera
from .configuration_photoneo import PhotoneoCameraConfig


class PhotoneoCamera(Camera):  # ✅ 继承 Camera 基类
    """
    Photoneo PhoXi 3D 扫描仪包装器
    基于 Harvester SDK (GenICam)
    """
    def __init__(self, config: PhotoneoCameraConfig):
        """
        Args:
            config: PhotoneoCameraConfig 配置对象
        """
        super().__init__(config)  # ✅ 调用基类构造函数
        
        # 从配置对象提取参数
        dev_id = config.device_id
        camera_translation = config.translation
        camera_quaternion = config.quaternion
        external_calibration_path = config.calibration_path
        
        # 设备 ID
        self.device_id = f"PhotoneoTL_DEV_{dev_id}"
        logger.info(f"Initializing Photoneo: {self.device_id}")
        
        # CTI 文件路径
        if os.getenv('PHOXI_CONTROL_PATH') is not None:
            self.cti_file_path = os.path.join(
                os.getenv('PHOXI_CONTROL_PATH'), 
                "API/lib/photoneo.cti"
            )
        else:
            self.cti_file_path = "/opt/Photoneo/PhoXiControl-1.15.0/API/lib/photoneo.cti"
        
        if not os.path.exists(self.cti_file_path):
            raise FileNotFoundError(
                f"❌ Photoneo CTI file not found: {self.cti_file_path}\n"
                f"Please install PhoXi Control or set PHOXI_CONTROL_PATH env"
            )
        
        logger.info(f"CTI path: {self.cti_file_path}")
        
        # 初始化外参变换
        self.camera_to_world_T = None
        
        if camera_translation is not None and camera_quaternion is not None:
            self._set_extrinsics_from_pose(camera_translation, camera_quaternion)
        elif external_calibration_path and os.path.exists(external_calibration_path):
            self.load_extrinsics(external_calibration_path)
        else:
            logger.warning("⚠️  未设置相机外参，点云将保持在相机坐标系")
        
        # 初始化 Harvester
        self.h = None
        self.ia = None
        self.features = None
        self._is_connected = False
        
        # 异步读取相关
        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: NDArray[Any] | None = None
        self.new_frame_event: Event = Event()

    @property
    def is_connected(self) -> bool:
        """检查相机是否已连接"""
        return self._is_connected
    
    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """检测可用的 Photoneo 设备"""
        found_cameras = []
        
        try:
            from harvesters.core import Harvester
            
            if os.getenv('PHOXI_CONTROL_PATH') is not None:
                cti_file_path = os.path.join(
                    os.getenv('PHOXI_CONTROL_PATH'), 
                    "API/lib/photoneo.cti"
                )
            else:
                cti_file_path = "/opt/Photoneo/PhoXiControl-1.15.0/API/lib/photoneo.cti"
            
            if not os.path.exists(cti_file_path):
                logger.warning(f"❌ CTI file not found: {cti_file_path}")
                return found_cameras
            
            h = Harvester()
            h.add_file(cti_file_path, check_existence=True, check_validity=True)
            h.update()
            
            for item in h.device_info_list:
                found_cameras.append({
                    'name': f"{item.property_dict.get('vendor', 'Unknown')} {item.property_dict.get('model', 'Unknown')}",
                    'id': item.property_dict.get('id_', 'Unknown'),
                    'serial_number': item.property_dict.get('serial_number', 'Unknown')
                })
            
            h.reset()
        
        except Exception as e:
            logger.error(f"Failed to find Photoneo cameras: {e}")
        
        return found_cameras

    def _set_extrinsics_from_pose(self, translation, quaternion):
        """从位姿参数设置外参变换"""
        try:
            from scipy.spatial.transform import Rotation as R_scipy
        except ImportError:
            raise ImportError("需要 scipy 来处理四元数，请安装: pip install scipy")
        
        rotation = R_scipy.from_quat(quaternion)
        R_cam_to_world = rotation.as_matrix()
        
        self.camera_to_world_T = np.eye(4, dtype=np.float32)
        self.camera_to_world_T[:3, :3] = R_cam_to_world
        self.camera_to_world_T[:3, 3] = translation
        
        logger.success("✅ 相机外参已设置 (从位姿参数)")
        logger.info(f"   平移: {translation}")
        logger.info(f"   四元数: {quaternion}")
    
    def load_extrinsics(self, txt_path):
        """从标定文件加载相机外参"""
        try:
            with open(txt_path, 'r') as f:
                f.readline()  # skip intrinsics
                f.readline()  # skip distortion
                num_list = f.readline().split(' ')[:-1]
            
            R_cam_to_world = np.array([float(x) for x in num_list[:9]]).reshape(3, 3)
            t_cam_to_world = np.array([float(x) for x in num_list[9:12]])
            
            self.camera_to_world_T = np.eye(4, dtype=np.float32)
            self.camera_to_world_T[:3, :3] = R_cam_to_world
            self.camera_to_world_T[:3, 3] = t_cam_to_world
            
            logger.success(f"✅ 相机外参已加载 (从文件: {txt_path})")
            
        except Exception as e:
            logger.error(f"❌ 加载外参文件失败: {e}")
            self.camera_to_world_T = None

    def connect(self, warmup: bool = True) -> None:
        """连接相机"""
        if self.is_connected:
            logger.warning(f"{self} already connected")
            return
        
        try:
            from harvesters.core import Harvester
        except ImportError:
            raise ImportError(
                "❌ harvesters not installed! Install with:\n"
                "pip install harvesters"
            )
        
        # 如果 Harvester 已存在，先清理
        if self.h is not None:
            try:
                self.h.reset()
            except Exception as e:
                logger.warning(f"Failed to reset existing Harvester: {e}")
            self.h = None
    
        # 创建新的 Harvester 实例
        self.h = Harvester()
        self.h.add_file(self.cti_file_path, check_existence=True, check_validity=True)
        self.h.update()
        
        logger.info("Available Photoneo devices:")
        for item in self.h.device_info_list:
            logger.info(f"  - {item.property_dict['serial_number']}: {item.property_dict['id_']}")
        
        # ✅ 尝试连接设备（添加重试机制）
        max_retries = 5  # 增加重试次数
        retry_delay = 2.0  # 增加延迟到 2 秒
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Connection attempt {attempt + 1}/{max_retries}...")
                self.ia = self.h.create({'id_': self.device_id})
                
                # ✅ 验证设备是否真正可用（尝试读取一个简单的属性）
                self.features = self.ia.remote_device.node_map
                
                # 尝试配置设备，如果失败说明设备还没准备好
                try:
                    trigger_mode = self.features.PhotoneoTriggerMode.value
                    logger.info(f"Device ready, current TriggerMode: {trigger_mode}")
                    break  # 设备可用，跳出循环
                except Exception as e:
                    logger.warning(f"Device not ready: {e}")
                    # 销毁这次连接，准备重试
                    try:
                        self.ia.destroy()
                    except:
                        pass
                    self.ia = None
                    self.features = None
                    
                    if attempt < max_retries - 1:
                        logger.info(f"Waiting {retry_delay}s before retry...")
                        time.sleep(retry_delay)
                    else:
                        raise RuntimeError(f"Device not ready after {max_retries} attempts")
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Connection failed: {e}")
                    logger.info(f"Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    raise  # 最后一次尝试失败，抛出异常
    
        # ✅ 配置触发模式
        try:
            logger.info(f"TriggerMode before: {self.features.PhotoneoTriggerMode.value}")
            self.features.PhotoneoTriggerMode.value = "Software"
            logger.info(f"TriggerMode after: {self.features.PhotoneoTriggerMode.value}")
        except Exception as e:
            logger.error(f"Failed to configure TriggerMode: {e}")
            raise  # 配置失败应该抛出异常，而不是只警告
    
        # ✅ 使能输出结构
        try:
            self.features.SendTexture.value = False
            self.features.SendPointCloud.value = True
            self.features.SendNormalMap.value = False
            self.features.SendDepthMap.value = False
            self.features.SendConfidenceMap.value = False
        except Exception as e:
            logger.error(f"Failed to configure output structures: {e}")
            raise  # 配置失败应该抛出异常
    
        self._is_connected = True
        logger.success(f"✅ {self} connected")

    def read(self, color_mode=None) -> NDArray[Any]:
        """同步读取点云（阻塞）"""
        if not self.is_connected:
            raise RuntimeError(f"{self} is not connected")
        
        try:
            self.ia.stop()
            self.ia.start()
            
            self.features.TriggerFrame.execute()
            buffer = self.ia.fetch(timeout=5.0)
            
            point_cloud_component = buffer.payload.components[2]
            
            if point_cloud_component.width == 0 or point_cloud_component.height == 0:
                logger.warning("Empty point cloud captured")
                return np.zeros((0, 3), dtype=np.float32)
            
            point_cloud_cam = point_cloud_component.data.reshape(
                point_cloud_component.height * point_cloud_component.width, 3
            ).copy()
            
            # mm -> m
            point_cloud_cam = point_cloud_cam / 1000.0
            
            # 移除无效点
            valid_mask = np.linalg.norm(point_cloud_cam, axis=1) > 1e-6
            point_cloud_cam = point_cloud_cam[valid_mask]
            
            # 应用外参变换
            if self.camera_to_world_T is not None:
                ones = np.ones((len(point_cloud_cam), 1), dtype=np.float32)
                point_cloud_homo = np.hstack([point_cloud_cam, ones])
                point_cloud_world_homo = (self.camera_to_world_T @ point_cloud_homo.T).T
                return point_cloud_world_homo[:, :3]
            else:
                return point_cloud_cam
            
        except Exception as e:
            logger.error(f"Failed to read point cloud: {e}")
            return np.zeros((0, 3), dtype=np.float32)
    
    def _read_loop(self) -> None:
        """后台线程循环"""
        if self.stop_event is None:
            raise RuntimeError("stop_event not initialized")
        
        while not self.stop_event.is_set():
            try:
                frame = self.read()
                with self.frame_lock:
                    self.latest_frame = frame
                self.new_frame_event.set()
            except Exception as e:
                logger.error(f"Error in read loop: {e}")
                time.sleep(0.1)
    
    def _start_read_thread(self) -> None:
        """启动后台读取线程"""
        if self.thread is not None and self.thread.is_alive():
            logger.warning(f"{self} read thread already running")
            return
        if self.stop_event is not None:
            self.stop_event.clear()
        
        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, name=f"{self}_read_loop")
        self.thread.daemon = True
        self.thread.start()
        logger.info(f"{self} async read thread started")
    
    def _stop_read_thread(self) -> None:
        """停止后台读取线程"""
        if self.stop_event is not None:
            self.stop_event.set()
        
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        
        self.thread = None
        self.stop_event = None
    
    def async_read(self, timeout_ms: float = 1000) -> NDArray[Any]:
        """异步读取最新点云（非阻塞）"""
        if not self.is_connected:
            raise RuntimeError(f"{self} is not connected")
        
        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()
        
        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(f"Timeout waiting for frame from {self}")
        
        with self.frame_lock:
            frame = self.latest_frame
        
        if frame is None:
            raise RuntimeError(f"No frame available from {self}")
        
        return frame
    
    def disconnect(self) -> None:
        """断开相机连接"""
        if not self.is_connected and self.thread is None:
            logger.warning(f"{self} already disconnected")
            return
        
        # ✅ 先停止后台线程
        if self.thread is not None:
            logger.info("Stopping background thread...")
            self._stop_read_thread()
        
        # ✅ 立即设置断开状态（避免异常导致状态不一致）
        self._is_connected = False
        
        try:
            # 停止采集流
            if self.ia is not None:
                try:
                    logger.info("Stopping acquisition...")
                    self.ia.stop()
                except Exception as e:
                    logger.warning(f"Error stopping acquisition: {e}")
                
                try:
                    logger.info("Destroying image acquirer...")
                    self.ia.destroy()
                except Exception as e:
                    logger.warning(f"Error destroying image acquirer: {e}")
                finally:
                    self.ia = None
            
            # 重置 Harvester
            if self.h is not None:
                try:
                    logger.info("Resetting Harvester...")
                    self.h.reset()
                except Exception as e:
                    logger.warning(f"Error resetting Harvester: {e}")
                finally:
                    self.h = None
            
            self.features = None
            logger.info(f"{self} hardware disconnected")
            
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
            # ✅ 即使出错也保持断开状态
    
    def __str__(self) -> str:
        return f"PhotoneoCamera({self.device_id})"

    def __del__(self):
        """析构时确保资源释放"""
        if self.is_connected:
            self.disconnect()
