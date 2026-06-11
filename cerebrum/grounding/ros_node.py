import sys
import numpy as np
from ..types import Exogenous

# We need to import our grounding components
from .sensory import SensoryProcessor
from .motor import MotorProcessor

# Try to import real rclpy
try:
    import rclpy
    from rclpy.node import Node as ROSNode
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    
# Import message types
try:
    from std_msgs.msg import Float64MultiArray as ROSFloat64MultiArray
    STD_MSGS_AVAILABLE = True
except ImportError:
    STD_MSGS_AVAILABLE = False

# Fallback definition for Mock classes
class MockPublisher:
    _queue = []
    _processing = False

    def __init__(self, topic_name, msg_type):
        self.topic_name = topic_name
        self.msg_type = msg_type
        self.published_messages = []
        
    def publish(self, msg):
        self.published_messages.append(msg)
        MockPublisher._queue.append((self, msg))
        if not MockPublisher._processing:
            MockPublisher._processing = True
            try:
                while MockPublisher._queue:
                    pub, current_msg = MockPublisher._queue.pop(0)
                    if pub.topic_name in MockRclpy.subscriptions:
                        for sub in MockRclpy.subscriptions[pub.topic_name]:
                            sub.callback(current_msg)
            finally:
                MockPublisher._processing = False

class MockSubscription:
    def __init__(self, topic_name, msg_type, callback):
        self.topic_name = topic_name
        self.msg_type = msg_type
        self.callback = callback

class MockLogger:
    def info(self, msg):  pass
    def warn(self, msg):  pass
    def error(self, msg): pass

class MockNode:
    def __init__(self, node_name):
        self.node_name = node_name
        self.publishers = {}
        self.subscriptions = {}
        self.logger = MockLogger()
        
    def create_publisher(self, msg_type, topic_name, qos_profile=10):
        pub = MockPublisher(topic_name, msg_type)
        self.publishers[topic_name] = pub
        MockRclpy.publishers.setdefault(topic_name, []).append(pub)
        return pub
        
    def create_subscription(self, msg_type, topic_name, callback, qos_profile=10):
        sub = MockSubscription(topic_name, msg_type, callback)
        self.subscriptions[topic_name] = sub
        MockRclpy.subscriptions.setdefault(topic_name, []).append(sub)
        return sub
        
    def get_logger(self):
        return self.logger
        
    def destroy_node(self):
        pass

class MockRclpy:
    publishers = {}
    subscriptions = {}
    initialized = False
    
    @classmethod
    def init(cls, args=None):
        cls.publishers.clear()
        cls.subscriptions.clear()
        cls.initialized = True
        MockPublisher._queue = []
        MockPublisher._processing = False
        
    @classmethod
    def shutdown(cls):
        cls.publishers.clear()
        cls.subscriptions.clear()
        cls.initialized = False
        MockPublisher._queue = []
        MockPublisher._processing = False
        
    @classmethod
    def create_node(cls, node_name):
        if not cls.initialized:
            raise RuntimeError("rclpy not initialized")
        return MockNode(node_name)
        
    @classmethod
    def spin_once(cls, node, timeout_sec=0.0):
        import time
        time.sleep(timeout_sec)

class std_msgs:
    class msg:
        class Float64MultiArray:
            def __init__(self):
                self.data = []

# Register mocks in sys.modules if real ones not present
if not ROS_AVAILABLE:
    sys.modules["rclpy"] = MockRclpy
    RclpyClass = MockRclpy
    NodeClass = MockNode
else:
    sys.modules.setdefault("rclpy", rclpy)
    RpyClass = rclpy
    NodeClass = ROSNode

if not STD_MSGS_AVAILABLE:
    sys.modules["std_msgs"] = std_msgs
    sys.modules["std_msgs.msg"] = std_msgs.msg
    Float64MultiArrayClass = std_msgs.msg.Float64MultiArray
else:
    sys.modules.setdefault("std_msgs", sys.modules.get("std_msgs"))
    Float64MultiArrayClass = ROSFloat64MultiArray

class CerebrumROSNode(NodeClass):
    def __init__(self, net, node_name="cerebrum_ros_node", reflex=None, sensory_processor=None, motor_processor=None):
        super().__init__(node_name)
        import threading
        self._lock = threading.RLock()
        self.net = net
        self.reflex = reflex
        self.sensory_processor = sensory_processor or SensoryProcessor()
        self.motor_processor = motor_processor or MotorProcessor()
        self.reward = 1.0  # Default initial reward
        
        # Publishers
        self.motor_pub = self.create_publisher(Float64MultiArrayClass, "/motor_commands")
        self.telemetry_pub = self.create_publisher(Float64MultiArrayClass, "/telemetry")
        
        # Subscriptions
        self.sensory_sub = self.create_subscription(
            Float64MultiArrayClass,
            "/sensory_input",
            self.sensory_callback
        )
        self.reward_sub = self.create_subscription(
            Float64MultiArrayClass,
            "/reward",
            self.reward_callback
        )
        
    def reward_callback(self, msg):
        with self._lock:
            try:
                if msg is None or not hasattr(msg, 'data'):
                    self.get_logger().warn("Malformed reward message: msg has no data attribute.")
                    return
                if len(msg.data) > 0:
                    val = float(msg.data[0])
                    if np.isnan(val) or np.isinf(val):
                        self.get_logger().warn("NaN/Inf received in reward callback, skipping.")
                        return
                    self.reward = val
            except (TypeError, ValueError) as e:
                self.get_logger().error(f"Error processing reward message: {e}")
            
    def sensory_callback(self, msg):
        with self._lock:
            try:
                if msg is None or not hasattr(msg, 'data'):
                    self.get_logger().warn("Malformed sensory message: msg has no data attribute.")
                    return
                
                # Validate and clean NaN/Inf sensory inputs
                cleaned_data = []
                has_invalid = False
                for x in msg.data:
                    val = float(x)
                    if np.isnan(val) or np.isinf(val):
                        cleaned_data.append(0.0)
                        has_invalid = True
                    else:
                        cleaned_data.append(val)
                if has_invalid:
                    self.get_logger().warn("NaN/Inf detected in sensory input; replacing with 0.0.")
                msg.data = cleaned_data

                data_len = len(msg.data)
                M_ = self.net.M_
                slice_dim = self.net.modules[0].cfg.dims[0]
                
                bypass_active = False
                action_u = None
                
                if self.reflex is not None:
                    if data_len == 5:
                        state = np.asarray(msg.data, dtype=float)
                        # Construct a dictionary matching the semantic positional mapping:
                        # index 0: dist, index 1: tilt, index 2: error_energy
                        state_to_evaluate = {
                            "dist": float(state[0]),
                            "tilt": float(state[1]),
                            "error_energy": float(state[2])
                        }
                    else:
                        if data_len >= 8:
                            lidar = msg.data[:4]
                            camera = msg.data[4:6]
                            odometer = msg.data[6:8]
                        else:
                            lidar = msg.data
                            camera = []
                            odometer = []
                        state = self.sensory_processor.process(lidar, camera, odometer)
                        # Construct a dictionary with the correct mapping from processed state:
                        # state[0] = min_lidar (dist)
                        # state[1] = left_cam (camera, NOT tilt)
                        # state[2] = right_cam (camera, NOT error_energy)
                        # state[3] = velocity
                        # state[4] = heading
                        state_to_evaluate = {
                            "dist": float(state[0]),
                            "tilt": 0.0,
                            "error_energy": 0.0
                        }
                        
                    bypass_active, action_u = self.reflex.evaluate(state_to_evaluate)
                    
                if bypass_active and action_u is not None:
                    cmd_msg = Float64MultiArrayClass()
                    cmd_msg.data = action_u.tolist()
                    self.motor_pub.publish(cmd_msg)
                    
                    telem_msg = Float64MultiArrayClass()
                    telem_msg.data = [1.0, 0.0]
                    self.telemetry_pub.publish(telem_msg)
                else:
                    obs_slices = []
                    expected_len = M_ * slice_dim
                    if data_len >= expected_len:
                        for i in range(M_):
                            obs_slices.append(np.array(msg.data[i*slice_dim : (i+1)*slice_dim]))
                    else:
                        flat_data = np.zeros(expected_len)
                        n_copy = min(data_len, expected_len)
                        flat_data[:n_copy] = msg.data[:n_copy]
                        for i in range(M_):
                            obs_slices.append(flat_data[i*slice_dim : (i+1)*slice_dim])
                    
                    action = Exogenous(np.zeros(2))
                    
                    z, M_val = self.net.step(obs_slices, action, reward=self.reward)
                    
                    action_vector = z[:, 0] if z.ndim > 1 else z
                    vels = self.motor_processor.process(action_vector)
                    
                    cmd_msg = Float64MultiArrayClass()
                    cmd_msg.data = vels.tolist()
                    self.motor_pub.publish(cmd_msg)
                    
                    telem_msg = Float64MultiArrayClass()
                    telem_msg.data = [2.0, float(M_val)] + z.flatten().tolist()
                    self.telemetry_pub.publish(telem_msg)
            except (TypeError, ValueError) as e:
                self.get_logger().error(f"Error processing sensory message: {e}")
