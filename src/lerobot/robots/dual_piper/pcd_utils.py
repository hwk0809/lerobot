import numpy as np
import open3d as o3d
import torch
from loguru import logger
from scipy.spatial.transform import Rotation

def process_point_cloud(raw_pcd, num_points=2048, use_gpu=True, device="cuda", visualize=False) -> np.ndarray:
    """
    1. Workspace Crop
    2. Voxel Downsample - if too many points
    3. Far Point Sample (FPS) - unify to num_points
    """
    if raw_pcd is None or len(raw_pcd) == 0: 
        return np.zeros((num_points, 3), dtype=np.float32)
    
    # save_raw_pcd = raw_pcd.copy()   
    # o3d.io.write_point_cloud("photoneo_raw.pcd", o3d.geometry.PointCloud(o3d.utility.Vector3dVector(save_raw_pcd)))
    # o3d.io.read_point_cloud("photoneo_raw.pcd")
    # o3d.visualization.draw_geometries(
    #     [o3d.geometry.PointCloud(o3d.utility.Vector3dVector(save_raw_pcd))],
    #     window_name="Raw Point Cloud",
    #     width=1024,
    #     height=768,
    #     point_show_normal=False
    # )
    # 1. Workspace Crop, filter the grond(actually do not filter ground is ok)
    # cropper the table area
    mask = (raw_pcd[:, 2] > 0.0015) & (raw_pcd[:, 2] < 1.5)
    # mask &= (np.abs(raw_pcd[:, 0]) < 1.0) & (np.abs(raw_pcd[:, 1]) < 1.0)
    
    # cropper the workspace area
    mask &= (np.abs(raw_pcd[:, 0]) < 0.63) & (np.abs(raw_pcd[:, 1]) < 0.5)

    # cropper the arm base area
    # mask &= ~ ((raw_pcd[:, 0] > -0.05) & (raw_pcd[:, 0] < 0.08) & (raw_pcd[:,2] < 0.025))
    mask &= ~ (abs(raw_pcd[:, 0] < 0.1))    


    # cropper the mess area   
    mask &= ~ ( (abs(raw_pcd[:, 1]) > 0.35) &  (raw_pcd[:,2] < 0.03))
    # mask &= ~ ((raw_pcd[:, 0] < 0.05) &  (raw_pcd[:,2] < 0.02))
    # mask &= ~ ((raw_pcd[:, 0] > 0.4) &  (np.abs(raw_pcd[:,1]) > 0.4))
               
    # mask &= ~ ((raw_pcd[:, 0] > -0.05) &  (np.abs(raw_pcd[:,2]) > 0.25) &  (raw_pcd[:,2] < 0.))
    filtered_pcd = raw_pcd[mask]

    if visualize:
        visualize_single_pcd(raw_pcd, "1. Raw Point Cloud")
        visualize_single_pcd(filtered_pcd, "2. After Crop")
    
    if len(filtered_pcd) == 0: 
        return np.zeros((num_points, 3), dtype=np.float32)
    
    # 2. 离群点过滤
    if len(filtered_pcd) > 50:
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(filtered_pcd)
        pcd_o3d, _ = pcd_o3d.remove_statistical_outlier(nb_neighbors=30, std_ratio=3.0)
        filtered_pcd = np.asarray(pcd_o3d.points)
        
        if visualize:
            visualize_single_pcd(filtered_pcd, "3. After Statistical Outlier Removal")



    # 2. Voxel Downsample (if number points > 2*num_points，accelerate FPS)
    if len(filtered_pcd) > 2*num_points:
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(filtered_pcd)
        pcd_o3d = pcd_o3d.voxel_down_sample(voxel_size=0.01)
        filtered_pcd = np.asarray(pcd_o3d.points)

    # 3. FPS sample
    output_pcd = None
    if use_gpu and torch.cuda.is_available():
        down_sampled_pcd = _fps_torch(filtered_pcd, num_points, device=device)
        if visualize:
            visualize_single_pcd(down_sampled_pcd, "4. After FPS (GPU)")
        return down_sampled_pcd
    else:
        down_sampled_pcd = _fps_open3d(filtered_pcd, num_points)
        if visualize:
            visualize_single_pcd(down_sampled_pcd, "4. After FPS (CPU)")
        return down_sampled_pcd

def _fps_torch(points: np.ndarray, n_samples: int, device="cuda") -> np.ndarray:
    """
    GPU  Farthest Point Sampling
    """
    N = len(points)
    
    # Case 1: Not enough points, randomly upsample
    if N < n_samples:
        indices = np.random.choice(N, n_samples, replace=True)
        return points[indices]
    
    # Case 2: Just enough points
    if N == n_samples:
        return points.copy()
    
    # Case 3: FPS downsample
    points_tensor = torch.from_numpy(points).float().to(device)
    
    sample_inds = torch.zeros(n_samples, dtype=torch.long, device=device)
    distances = torch.full((N,), float('inf'), device=device)
    
    # randomly select the first point
    start_idx = torch.randint(0, N, (1,), device=device)
    sample_inds[0] = start_idx
    
    for i in range(1, n_samples):
        last_idx = sample_inds[i - 1]
        dist = torch.sum((points_tensor - points_tensor[last_idx]) ** 2, dim=-1)
        distances = torch.min(distances, dist)
        sample_inds[i] = torch.argmax(distances)
    
    return points_tensor[sample_inds].cpu().numpy()

def _fps_open3d(points: np.ndarray, n_samples: int) -> np.ndarray:
    """
    CPU fallback
    """
    N = len(points)
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(points)
    
    if N >= n_samples:
        pcd_o3d = pcd_o3d.farthest_point_down_sample(n_samples)
        res = np.asarray(pcd_o3d.points)
        # Sometimes open3d FPS returns fewer points, need to pad
        if len(res) < n_samples:
             diff = n_samples - len(res)
             idxs = np.random.choice(len(res), diff)
             res = np.concatenate([res, res[idxs]])
        return res
    else:
        # 上采样
        choice = np.random.choice(N, n_samples - N)
        return np.concatenate([points, points[choice]])

def visualize_single_pcd(points, title="Point Cloud"):
    """简单的 Open3D 可视化"""
    if len(points) == 0:
        logger.warning(f"⚠️  {title}: 点云为空")
        return
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 根据高度着色
    colors = np.zeros_like(points)
    z_norm = (points[:, 2] - points[:, 2].min()) / (points[:, 2].max() - points[:, 2].min() + 1e-6)
    colors[:, 0] = z_norm
    colors[:, 1] = 1 - z_norm
    colors[:, 2] = 0.5
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 添加坐标系
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    
    logger.info(f" {title} ({len(points)} points) - 关闭窗口继续...")
    o3d.visualization.draw_geometries(
        [pcd, coord],
        window_name=title,
        width=1024,
        height=768,
        point_show_normal=False
    )


class NonBlockingVisualizer:
    """非阻塞式 Open3D 可视化器，避免死机"""
    def __init__(self, window_name="Point Cloud", width=1024, height=768,
                 camera_pos=None,camera_quat=None, camera_lookat=None, camera_up=None):
        self.vis = None
        self.window_name = window_name
        self.width = width
        self.height = height
        self.pcd = o3d.geometry.PointCloud()
        self.coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.3, origin=[0, 0, 0]
        )
        self.initialized = False

        self.camera_pos = camera_pos if camera_pos is not None else [-0.03846401, -0.11231157, 1.13300097]
        self.camera_quat = camera_quat if camera_quat is not None else [0.70629295, -0.69512124, 0.09361616, -0.09587879]
        self.camera_lookat = camera_lookat if camera_lookat is not None else [0.0, 0.0, 0.0]
        self.camera_up = camera_up if camera_up is not None else [0.0, 0.0, 1.0]
    
    def update(self, points):
        """更新点云（非阻塞）"""
        try:
            if not self.initialized:
                self.vis = o3d.visualization.Visualizer()
                self.vis.create_window(
                    window_name=self.window_name,
                    width=self.width,
                    height=self.height
                )
                self.vis.add_geometry(self.pcd)
                self.vis.add_geometry(self.coord_frame)
                self._set_camera_view() 
                self.initialized = True
            
            self.pcd.points = o3d.utility.Vector3dVector(points)
            self.vis.update_geometry(self.pcd)
            self.vis.poll_events()
            self.vis.update_renderer()
            return True
        except Exception as e:
            logger.warning(f"可视化更新失败: {e}")
            return False
    
    def close(self):
        if self.vis is not None:
            try:
                self.vis.destroy_window()
            except:
                pass
            self.vis = None
            self.initialized = False
    
    def _set_camera_view(self):
        """设置相机视角"""
        try:
            # 获取视图控制器
            ctr = self.vis.get_view_control()

            # 方法 1: 使用四元数设置
            
            # scene_center = np.array([0.0, 0.0, 0.02])

            rot = Rotation.from_quat(self.camera_quat)  # [x, y, z, w] 格式
            R = rot.as_matrix()

            camera_forward = R @ np.array([0, 0, 1])   # Z轴：相机朝向
            camera_up = -R @ np.array([0, 1, 0]) 

            lookat_distance = 0.5  # 观察距离（米），可调整
            lookat_point = self.camera_pos + camera_forward * lookat_distance
            
            ctr.set_lookat(lookat_point.tolist())
            ctr.set_front((-camera_forward).tolist())  # Open3D 使用相反方向
            ctr.set_up(camera_up.tolist())
            ctr.set_zoom(0.7)  # 缩放比例，可调整
            
            logger.info(f"✅ 相机视角已设置")
            logger.debug(f"   位置: {self.camera_pos}")
            logger.debug(f"   朝向: {camera_forward}")
            logger.debug(f"   上方向: {camera_up}")
            
 
            
        except Exception as e:
            logger.warning(f"设置相机视角失败: {e}")