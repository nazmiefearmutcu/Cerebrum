import numpy as np

class SensoryProcessor:
    """Transforms raw sensor readings into a normalized 5-dimensional workspace state."""
    def process(self, lidar_data, camera_data, odometer_data):
        # 1. Protection against None, NaN, and Inf
        if lidar_data is None:
            lidar_data = np.array([])
        else:
            lidar_data = np.asarray(lidar_data, dtype=float)
            lidar_data = np.where(np.isnan(lidar_data) | np.isinf(lidar_data), 10.0, lidar_data)
            
        if camera_data is None:
            camera_data = np.array([])
        else:
            camera_data = np.asarray(camera_data, dtype=float)
            camera_data = np.where(np.isnan(camera_data) | np.isinf(camera_data), 0.0, camera_data)
            
        if odometer_data is None:
            odometer_data = np.array([])
        else:
            odometer_data = np.asarray(odometer_data, dtype=float)
            odometer_data = np.where(np.isnan(odometer_data) | np.isinf(odometer_data), 0.0, odometer_data)
        
        # 2. Extract min lidar
        min_lidar = float(np.min(lidar_data)) if len(lidar_data) > 0 else 1.0
        
        # 3. Visual splits
        left_slice = camera_data[:len(camera_data)//2] if len(camera_data) > 0 else np.array([])
        right_slice = camera_data[len(camera_data)//2:] if len(camera_data) > 0 else np.array([])
        left_cam = float(np.mean(left_slice)) if len(left_slice) > 0 else 0.0
        right_cam = float(np.mean(right_slice)) if len(right_slice) > 0 else 0.0
        
        # 4. Odometry
        velocity = float(odometer_data[0]) if len(odometer_data) > 0 else 0.0
        heading = float(odometer_data[1]) if len(odometer_data) > 1 else 0.0
        
        # 5. Clamping
        min_lidar = np.clip(min_lidar, 0.0, 10.0)
        left_cam = np.clip(left_cam, 0.0, 1.0)
        right_cam = np.clip(right_cam, 0.0, 1.0)
        
        return np.array([min_lidar, left_cam, right_cam, velocity, heading], dtype=float)
