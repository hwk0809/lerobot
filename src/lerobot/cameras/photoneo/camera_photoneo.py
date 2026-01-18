
from loguru import logger
import numpy as np
import time
import os
from threading import Thread, Event, Lock
from typing import Any
from numpy.typing import NDArray

from ..camera import Camera
from .configuration_photoneo import PhotoneoCameraConfig
class PhotoneoCamera:
    """
    Photoneo PhoXi 3D 扫描仪包装器
    基于 Harvester SDK (GenICam)
    """
    def __init__(self, 
                 dev_id='2020-12-039-LC3',
                 external_calibration_path=None,
                 # ✅ 新增：直接传入位姿参数
                 camera_translation=None,  # [x, y, z] 米
                 camera_quaternion=None,   # [x, y, z, w]
                 width=640, 
                 height=480):
        """
        Args:
            dev_id: Photoneo 设备 ID (不含 'PhotoneoTL_DEV_' 前缀)
            external_calibration_path: 外参标定文件路径 (可选)
            camera_translation: 相机在世界坐标系的平移向量 [x, y, z] (米)
            camera_quaternion: 相机在世界坐标系的旋转四元数 [x, y, z, w]
            width, height: 占位参数，保持接口一致
        """
        # 设备 ID
        self.device_id = f"PhotoneoTL_DEV_{dev_id}"
        logger.info(f"Initializing Photoneo: {self.device_id}")
        
        # CTI 文件路径（GenICam 传输层接口）
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
        
        # ✅ 初始化外参变换
        self.camera_to_world_T = None  # 相机坐标系到世界坐标系的变换矩阵 (4x4)
        
        # 优先级：直接传入参数 > 标定文件
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
        
        #  ✅ 异步读取相关（与 OpenCVCamera 一致）
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
        """
        检测可用的 Photoneo 设备
        
        Returns:
            List[Dict]: 设备列表，每个字典包含设备信息
        """
        found_cameras = []
        
        try:
            from harvesters.core import Harvester
            
            # 尝试多个可能的 CTI 路径
            cti_paths = [
                "/opt/Photoneo/PhoXiControl-1.15.0/API/lib/photoneo.cti",
                os.path.join(os.getenv('PHOXI_CONTROL_PATH', ''), "API/lib/photoneo.cti"),
            ]
            
            for cti_path in cti_paths:
                if not os.path.exists(cti_path):
                    continue
                
                h = Harvester()
                h.add_file(cti_path)
                h.update()
                
                for device_info in h.device_info_list:
                    camera_info = {
                        "name": f"Photoneo {device_info.property_dict.get('model', 'Unknown')}",
                        "type": "Photoneo",
                        "id": device_info.property_dict.get('id_', 'Unknown'),
                        "serial_number": device_info.property_dict.get('serial_number', 'Unknown'),
                    }
                    found_cameras.append(camera_info)
                
                h.reset()
                break
        
        except Exception as e:
            logger.warning(f"Failed to detect Photoneo cameras: {e}")
        
        return found_cameras

    
    def _set_extrinsics_from_pose(self, translation, quaternion):
        """
        从位姿参数设置外参变换
        
        Args:
            translation: [x, y, z] 相机在世界坐标系的位置 (米)
            quaternion: [x, y, z, w] 相机在世界坐标系的旋转四元数
        """
        try:
            from scipy.spatial.transform import Rotation as R_scipy
        except ImportError:
            raise ImportError("需要 scipy 来处理四元数，请安装: pip install scipy")
        
        # 将四元数转换为旋转矩阵
        # 注意: scipy 使用 scalar-last (x, y, z, w) 格式
        rotation = R_scipy.from_quat(quaternion)  # [x, y, z, w]
        R_cam_to_world = rotation.as_matrix()
        
        # 构造 4x4 齐次变换矩阵
        self.camera_to_world_T = np.eye(4, dtype=np.float32)
        self.camera_to_world_T[:3, :3] = R_cam_to_world
        self.camera_to_world_T[:3, 3] = translation
        
        logger.success("✅ 相机外参已设置 (从位姿参数)")
        logger.info(f"   平移: {translation}")
        logger.info(f"   四元数: {quaternion}")
        logger.debug(f"   变换矩阵:\n{self.camera_to_world_T}")
    
    def load_extrinsics(self, txt_path):
        """
        从标定文件加载相机外参
        文件格式: 每行依次为 intrinsics, distortion, rotation(9), translation(3), resolution
        """
        try:
            with open(txt_path, 'r') as f:
                f.readline()  # skip intrinsics
                f.readline()  # skip distortion
                
                # 读取旋转矩阵 (3x3)
                num_list = f.readline().split(' ')[:-1]
                R_cam_to_world = np.array([float(num) for num in num_list], 
                                         dtype=np.float32).reshape(3, 3)
                
                # 读取平移向量 (mm -> m)
                num_list = f.readline().split(' ')[:-1]
                t_cam_to_world = np.array([float(num) for num in num_list], 
                                         dtype=np.float32) / 1000.0
            
            # 构造 4x4 变换矩阵
            self.camera_to_world_T = np.eye(4, dtype=np.float32)
            self.camera_to_world_T[:3, :3] = R_cam_to_world
            self.camera_to_world_T[:3, 3] = t_cam_to_world
            
            logger.success(f"✅ 相机外参已加载 (从文件: {txt_path})")
            logger.debug(f"   变换矩阵:\n{self.camera_to_world_T}")
            
        except Exception as e:
            logger.error(f"❌ 加载外参文件失败: {e}")
            self.camera_to_world_T = None
    

    def connect(self, warmup: bool = True) -> None:
        """
        连接相机
        
        Args:
            warmup: 是否执行预热采集（兼容接口，Photoneo 不需要预热）
        """
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
        
        self.h = Harvester()
        self.h.add_file(self.cti_file_path, check_existence=True, check_validity=True)
        self.h.update()
        
        logger.info("Available Photoneo devices:")
        for item in self.h.device_info_list:
            logger.info(f"  - {item.property_dict['serial_number']}: {item.property_dict['id_']}")
        
        # 连接指定设备
        self.ia = self.h.create({'id_': self.device_id})
        self.features = self.ia.remote_device.node_map
        
        # 配置触发模式
        logger.info(f"TriggerMode before: {self.features.PhotoneoTriggerMode.value}")
        self.features.PhotoneoTriggerMode.value = "Software"
        logger.info(f"TriggerMode after: {self.features.PhotoneoTriggerMode.value}")
        
        # 使能输出结构
        self.features.SendTexture.value = False
        self.features.SendPointCloud.value = True
        self.features.SendNormalMap.value = False
        self.features.SendDepthMap.value = False
        self.features.SendConfidenceMap.value = False
        
        self._is_connected = True
        logger.success(f"✅ {self} connected")
    
    def get_point_cloud(self):
        """
        采集点云并返回世界坐标系下的 (N, 3) numpy 数组
        """
        try:
            # 停止并重启采集流
            self.ia.stop()
            self.ia.start()
            
            # 软件触发一帧
            self.features.TriggerFrame.execute()
            buffer = self.ia.fetch(timeout=5.0)
            
            payload = buffer.payload
            
            # 获取点云 Component
            point_cloud_component = payload.components[2]
            
            if point_cloud_component.width == 0 or point_cloud_component.height == 0:
                logger.warning("Empty point cloud received!")
                return np.zeros((0, 3), dtype=np.float32)
            
            # Reshape 到 (N, 3)
            point_cloud_cam = point_cloud_component.data.reshape(
                point_cloud_component.height * point_cloud_component.width, 3
            ).copy()
            
            # ✅ Photoneo 输出单位是 mm，转换为 m
            point_cloud_cam = point_cloud_cam / 1000.0
            
            # ✅ 移除无效点 (0, 0, 0)
            valid_mask = np.linalg.norm(point_cloud_cam, axis=1) > 1e-6
            point_cloud_cam = point_cloud_cam[valid_mask]
            
            # ✅ 应用外参变换：相机坐标系 -> 世界坐标系
            if self.camera_to_world_T is not None:
                # 转换为齐次坐标 (N, 4)
                points_homogeneous = np.hstack([
                    point_cloud_cam, 
                    np.ones((point_cloud_cam.shape[0], 1), dtype=np.float32)
                ])
                
                # 应用变换矩阵
                points_world = (self.camera_to_world_T @ points_homogeneous.T).T
                
                # 转回笛卡尔坐标 (N, 3)
                point_cloud_world = points_world[:, :3]
                
                logger.debug(f"Captured {point_cloud_world.shape[0]} points (world frame)")
                return point_cloud_world.astype(np.float32)
            else:
                logger.warning("⚠️  未设置外参，返回相机坐标系点云")
                return point_cloud_cam.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Failed to capture point cloud: {e}")
            return np.zeros((0, 3), dtype=np.float32)
    

    def read(self, color_mode=None) -> NDArray[Any]:
        """
        同步读取点云（阻塞）
        
        Args:
            color_mode: 占位参数（兼容 Camera 接口，Photoneo 不使用）
        
        Returns:
            np.ndarray: 点云数据，shape (N, 3)，单位：米
        """
        if not self.is_connected:
            raise RuntimeError(f"{self} is not connected")
        
        start_time = time.perf_counter()
        
        try:
            # 停止并重启采集流
            self.ia.stop()
            self.ia.start()
            
            # 软件触发一帧
            self.features.TriggerFrame.execute()
            buffer = self.ia.fetch(timeout=5.0)
            
            # 获取点云 Component
            point_cloud_component = buffer.payload.components[2]
            
            if point_cloud_component.width == 0 or point_cloud_component.height == 0:
                logger.warning("Empty point cloud received!")
                return np.zeros((0, 3), dtype=np.float32)
            
            # Reshape 到 (N, 3)
            point_cloud_cam = point_cloud_component.data.reshape(
                point_cloud_component.height * point_cloud_component.width, 3
            ).copy()
            
            # Photoneo 输出单位是 mm，转换为 m
            point_cloud_cam = point_cloud_cam / 1000.0
            
            # 移除无效点 (0, 0, 0)
            valid_mask = np.linalg.norm(point_cloud_cam, axis=1) > 1e-6
            point_cloud_cam = point_cloud_cam[valid_mask]
            
            # 应用外参变换：相机坐标系 -> 世界坐标系
            if self.camera_to_world_T is not None:
                points_homogeneous = np.hstack([
                    point_cloud_cam, 
                    np.ones((point_cloud_cam.shape[0], 1), dtype=np.float32)
                ])
                points_world = (self.camera_to_world_T @ points_homogeneous.T).T
                result = points_world[:, :3].astype(np.float32)
            else:
                result = point_cloud_cam.astype(np.float32)
            
            read_duration_ms = (time.perf_counter() - start_time) * 1e3
            logger.debug(f"{self} read took: {read_duration_ms:.1f}ms, points: {result.shape[0]}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to capture point cloud: {e}")
            return np.zeros((0, 3), dtype=np.float32)
    
    def _read_loop(self) -> None:
        """
        后台线程循环（与 OpenCVCamera 一致）
        """
        if self.stop_event is None:
            raise RuntimeError(f"{self}: stop_event is not initialized")
        
        while not self.stop_event.is_set():
            try:
                point_cloud = self.read()  # ← 调用同步读取
                
                # 更新缓冲区（线程安全）
                with self.frame_lock:
                    self.latest_frame = point_cloud
                self.new_frame_event.set()  # ← 通知有新帧
                
            except Exception as e:
                logger.warning(f"Error reading point cloud in background thread: {e}")
                time.sleep(0.1)  # 出错后短暂休眠
    
    def _start_read_thread(self) -> None:
        """启动后台读取线程（与 OpenCVCamera 一致）"""
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.1)
        if self.stop_event is not None:
            self.stop_event.set()
        
        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, name=f"{self}_read_loop")
        self.thread.daemon = True
        self.thread.start()
        logger.info(f"{self} async read thread started")
    
    def _stop_read_thread(self) -> None:
        """停止后台读取线程（与 OpenCVCamera 一致）"""
        if self.stop_event is not None:
            self.stop_event.set()
        
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        
        self.thread = None
        self.stop_event = None
    
    def async_read(self, timeout_ms: float = 1000) -> NDArray[Any]:
        """
        异步读取最新点云（非阻塞）
        
        Args:
            timeout_ms: 超时时间（毫秒），默认 1000ms（点云采集慢）
        
        Returns:
            np.ndarray: 点云数据，shape (N, 3)
        """
        if not self.is_connected:
            raise RuntimeError(f"{self} is not connected")
        
        # 启动后台线程（如果未启动）
        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()
        
        # 等待新帧事件
        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            thread_alive = self.thread is not None and self.thread.is_alive()
            raise TimeoutError(
                f"Timed out waiting for point cloud from {self} after {timeout_ms} ms. "
                f"Read thread alive: {thread_alive}."
            )
        
        # 从缓冲区读取最新帧（线程安全）
        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()
        
        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no frame available for {self}.")
        
        return frame
    
    def disconnect(self) -> None:
        """断开相机连接"""
        if not self.is_connected and self.thread is None:
            logger.warning(f"{self} not connected")
            return
        
        # 停止后台线程
        if self.thread is not None:
            self._stop_read_thread()
        
        # 关闭硬件连接
        try:
            if self.ia is not None:
                self.ia.stop()
                self.ia.destroy()
            if self.h is not None:
                self.h.reset()
            logger.info(f"{self} disconnected")
        except Exception as e:
            logger.error(f"Error closing camera: {e}")
        
        self._is_connected = False
    
    def __str__(self) -> str:
        return f"PhotoneoCamera({self.device_id})"
    