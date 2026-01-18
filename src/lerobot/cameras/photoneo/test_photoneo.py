#!/usr/bin/env python3
"""
Photoneo 相机测试脚本
用途：测试 LeRobot 集成的 PhotoneoCamera 类
"""
import os
import sys
import time
import numpy as np
import argparse
from loguru import logger
from pathlib import Path

# 引入 LeRobot 路径
project_root = Path(__file__).resolve().parents[4]  # lerobot/src/lerobot/cameras/photoneo -> lerobot
sys.path.insert(0, str(project_root))

# 导入 LeRobot 相机类
from lerobot.cameras.photoneo.camera_photoneo import PhotoneoCamera
from lerobot.cameras.photoneo.configuration_photoneo import PhotoneoCameraConfig


def test_photoneo_basic(config: PhotoneoCameraConfig):
    """基础测试：连接、同步读取、断开"""
    logger.info("=" * 60)
    logger.info("测试 1: 基础功能 (同步读取)")
    logger.info("=" * 60)
    
    camera = PhotoneoCamera(config)
    
    try:
        # 1. 测试连接
        logger.info("\n[1/3] 测试连接...")
        camera.connect()
        assert camera.is_connected, "相机连接失败"
        logger.success("✅ 连接成功")
        
        # 2. 测试同步读取
        logger.info("\n[2/3] 测试同步读取 (3 次)...")
        for i in range(3):
            start_time = time.perf_counter()
            point_cloud = camera.read()
            read_time = (time.perf_counter() - start_time) * 1000
            
            logger.info(f"  Frame {i+1}:")
            logger.info(f"    点云数量: {len(point_cloud)}")
            logger.info(f"    读取耗时: {read_time:.1f} ms")
            logger.info(f"    帧率: {1000.0/read_time:.1f} FPS")
            
            if len(point_cloud) > 0:
                logger.info(f"    点云范围: X=[{point_cloud[:, 0].min():.3f}, {point_cloud[:, 0].max():.3f}] m")
                logger.info(f"             Y=[{point_cloud[:, 1].min():.3f}, {point_cloud[:, 1].max():.3f}] m")
                logger.info(f"             Z=[{point_cloud[:, 2].min():.3f}, {point_cloud[:, 2].max():.3f}] m")
            
            time.sleep(0.2)
        
        logger.success("✅ 同步读取测试通过")
        
        # 3. 测试断开
        logger.info("\n[3/3] 测试断开...")
        camera.disconnect()
        assert not camera.is_connected, "相机断开失败"
        logger.success("✅ 断开成功")
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        raise
    finally:
        if camera.is_connected:
            camera.disconnect()


def test_photoneo_async(config: PhotoneoCameraConfig):
    """异步测试：后台线程采集"""
    logger.info("\n" + "=" * 60)
    logger.info("测试 2: 异步读取")
    logger.info("=" * 60)
    
    camera = PhotoneoCamera(config)
    
    try:
        # 1. 连接
        logger.info("\n[1/3] 连接相机...")
        camera.connect()
        logger.success("✅ 连接成功")
        
        # 2. 测试异步读取
        logger.info("\n[2/3] 测试异步读取 (5 次)...")
        logger.info("  提示: 第一次会启动后台线程，可能较慢")
        
        for i in range(5):
            start_time = time.perf_counter()
            
            try:
                point_cloud = camera.async_read(timeout_ms=2000)  # 2秒超时
                read_time = (time.perf_counter() - start_time) * 1000
                
                logger.info(f"  Frame {i+1}:")
                logger.info(f"    点云数量: {len(point_cloud)}")
                logger.info(f"    读取耗时: {read_time:.1f} ms")
                
                if i == 0:
                    logger.info(f"    (首次包含线程启动时间)")
                
            except TimeoutError:
                logger.warning(f"  Frame {i+1}: 超时！")
            
            time.sleep(0.1)  # 短暂延迟
        
        logger.success("✅ 异步读取测试通过")
        
        # 3. 测试线程状态
        logger.info("\n[3/3] 检查后台线程状态...")
        assert camera.thread is not None, "后台线程未启动"
        assert camera.thread.is_alive(), "后台线程已停止"
        logger.success("✅ 后台线程运行正常")
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        raise
    finally:
        if camera.is_connected:
            camera.disconnect()


def test_photoneo_with_processing(config: PhotoneoCameraConfig, args):
    """完整测试：采集 + 点云处理"""
    logger.info("\n" + "=" * 60)
    logger.info("测试 3: 采集 + 点云处理流程")
    logger.info("=" * 60)
    
    camera = PhotoneoCamera(config)
    
    # 导入你的点云处理函数
    try:
        project_root_parent = Path(__file__).resolve().parents[5]  # 回到 deformable_bench
        sys.path.insert(0, str(project_root_parent))
        from common.vision_utils import process_point_cloud
        logger.success("✅ 成功导入 process_point_cloud")
    except ImportError as e:
        logger.warning(f"⚠️  无法导入 process_point_cloud: {e}")
        logger.info("  跳过点云处理步骤")
        process_point_cloud = None
    
    try:
        # 1. 连接
        camera.connect()
        logger.success("✅ 相机已连接")
        
        # 2. 完整流程测试
        logger.info(f"\n开始采集 {args.num_frames} 帧点云...")
        
        for i in range(args.num_frames):
            logger.info(f"\n--- Frame {i+1}/{args.num_frames} ---")
            
            # 计时
            start_time = time.perf_counter()
            
            # 使用异步读取
            raw_pcd = camera.async_read(timeout_ms=2000)
            capture_time = (time.perf_counter() - start_time) * 1000
            
            # 处理点云
            if process_point_cloud is not None:
                process_start = time.perf_counter()
                obs_pcd = process_point_cloud(
                    raw_pcd, 
                    num_points=2048,
                    use_gpu=True,
                    visualize=(args.visualize and i == 0)  # 只可视化第一帧
                )
                process_time = (time.perf_counter() - process_start) * 1000
            else:
                obs_pcd = raw_pcd
                process_time = 0.0
            
            total_time = (time.perf_counter() - start_time) * 1000
            
            # 输出统计
            logger.info(f"✅ 完成:")
            logger.info(f"   原始点数: {len(raw_pcd)}")
            if process_point_cloud is not None:
                logger.info(f"   处理后点数: {len(obs_pcd)}")
            logger.info(f"   采集耗时: {capture_time:.1f} ms")
            if process_point_cloud is not None:
                logger.info(f"   处理耗时: {process_time:.1f} ms")
            logger.info(f"   总耗时: {total_time:.1f} ms")
            logger.info(f"   等效帧率: {1000.0/total_time:.1f} FPS")
            
            # 点云统计
            if len(obs_pcd) > 0:
                logger.info(f"   点云范围:")
                logger.info(f"     X: [{obs_pcd[:, 0].min():.3f}, {obs_pcd[:, 0].max():.3f}] m")
                logger.info(f"     Y: [{obs_pcd[:, 1].min():.3f}, {obs_pcd[:, 1].max():.3f}] m")
                logger.info(f"     Z: [{obs_pcd[:, 2].min():.3f}, {obs_pcd[:, 2].max():.3f}] m")
            
            # 保存第一帧
            if args.save_path and i == 0:
                np.save(args.save_path, obs_pcd)
                logger.success(f"💾 点云已保存: {args.save_path}")
            
            # 延迟
            if i < args.num_frames - 1:
                time.sleep(args.delay)
        
        logger.success("\n✅ 完整流程测试通过!")
        
    except Exception as e:
        logger.error(f"❌ 测试失败: {e}")
        raise
    finally:
        if camera.is_connected:
            camera.disconnect()


def test_photoneo_find_cameras():
    """测试设备检测"""
    logger.info("\n" + "=" * 60)
    logger.info("测试 4: 设备检测")
    logger.info("=" * 60)
    
    cameras = PhotoneoCamera.find_cameras()
    
    if len(cameras) == 0:
        logger.warning("⚠️  未检测到 Photoneo 设备")
    else:
        logger.success(f"✅ 检测到 {len(cameras)} 个设备:")
        for i, cam in enumerate(cameras):
            logger.info(f"  [{i+1}] {cam['name']}")
            logger.info(f"      ID: {cam['id']}")
            logger.info(f"      SN: {cam['serial_number']}")


def main():
    parser = argparse.ArgumentParser(
        description="Photoneo 相机 LeRobot 集成测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
测试模式:
  --test basic       : 基础功能测试（连接、同步读取、断开）
  --test async       : 异步读取测试（后台线程）
  --test processing  : 完整流程测试（采集 + 点云处理）
  --test find        : 设备检测测试
  --test all         : 运行所有测试（默认）

示例:
  # 运行所有测试（使用默认外参）
  python test_photoneo.py
  
  # 只测试基础功能
  python test_photoneo.py --test basic
  
  # 自定义外参
  python test_photoneo.py --camera_pos 1.541 0.139 0.759 --camera_quat 0.585 0.606 -0.400 -0.362
  
  # 完整流程测试 + 可视化
  python test_photoneo.py --test processing --num_frames 5 --visualize --save_path test_pcd.npy
        """
    )
    
    # 测试模式
    parser.add_argument("--test", type=str, default="all",
                       choices=["basic", "async", "processing", "find", "all"],
                       help="测试模式")
    
    # 相机配置
    parser.add_argument("--photoneo_id", type=str, default="PAG-076",
                       help="Photoneo 设备 ID (不含前缀)")
    
    parser.add_argument("--calib_path", type=str, default=None,
                       help="外参标定文件路径 (txt 格式)")
    
    # ✅ 默认外参（你提供的默认值）
    parser.add_argument("--camera_pos", type=float, nargs=3, 
                       default=[1.54116268, 0.13879753, 0.75927529],
                       metavar=('X', 'Y', 'Z'),
                       help="相机在世界坐标系的位置 (米)")
    
    parser.add_argument("--camera_quat", type=float, nargs=4,
                       default=[0.58455770, 0.60577063, -0.40007590, -0.36231688],
                       metavar=('X', 'Y', 'Z', 'W'),
                       help="相机在世界坐标系的旋转四元数 [x, y, z, w]")
    
    # 测试参数
    parser.add_argument("--num_frames", type=int, default=3,
                       help="采集帧数（仅 processing 模式）")
    
    parser.add_argument("--delay", type=float, default=0.5,
                       help="帧间延迟 (秒)")
    
    parser.add_argument("--visualize", action="store_true",
                       help="使用 Open3D 可视化第一帧点云")
    
    parser.add_argument("--save_path", type=str, default=None,
                       help="保存第一帧点云的路径 (npy 格式)")
    
    args = parser.parse_args()
    
    # 创建配置
    config = PhotoneoCameraConfig(
        device_id=args.photoneo_id,
        fps=25,  # Photoneo 典型帧率
        translation=args.camera_pos,
        quaternion=args.camera_quat,
        calibration_path=args.calib_path,
    )
    
    logger.info("Photoneo 相机配置:")
    logger.info(f"  设备ID: {config.device_id}")
    logger.info(f"  外参平移: {config.translation}")
    logger.info(f"  外参四元数: {config.quaternion}")
    if config.calibration_path:
        logger.info(f"  标定文件: {config.calibration_path}")
    logger.info("")
    
    # 运行测试
    try:
        if args.test == "find" or args.test == "all":
            test_photoneo_find_cameras()
        
        if args.test == "basic" or args.test == "all":
            test_photoneo_basic(config)
        
        if args.test == "async" or args.test == "all":
            test_photoneo_async(config)
        
        if args.test == "processing" or args.test == "all":
            test_photoneo_with_processing(config, args)
        
        logger.success("\n" + "=" * 60)
        logger.success("🎉 所有测试通过!")
        logger.success("=" * 60)
        
    except Exception as e:
        logger.error("\n" + "=" * 60)
        logger.error(f"❌ 测试失败: {e}")
        logger.error("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()