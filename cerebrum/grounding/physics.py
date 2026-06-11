import sys
import numpy as np

# Try to import real pybullet
try:
    import pybullet as real_p
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False

class MockPyBullet:
    GUI = 1
    DIRECT = 2
    
    def __init__(self):
        self.connected = False
        self.gravity = [0.0, 0.0, -9.81]
        self.bodies = {}
        self.time_step = 1.0 / 240.0
        
    def connect(self, connection_mode=1):
        self.connected = True
        if PYBULLET_AVAILABLE:
            try:
                return real_p.connect(connection_mode)
            except Exception:
                pass
        return 0
        
    def disconnect(self):
        self.connected = False
        if PYBULLET_AVAILABLE:
            try:
                real_p.disconnect()
            except Exception:
                pass
        
    def setGravity(self, x, y, z):
        self.gravity = [x, y, z]
        if PYBULLET_AVAILABLE:
            try:
                real_p.setGravity(x, y, z)
            except Exception:
                pass
        
    def loadURDF(self, urdf_path, basePosition=(0.0, 0.0, 0.0), baseOrientation=(0.0, 0.0, 0.0, 1.0)):
        body_id = len(self.bodies) + 1
        self.bodies[body_id] = {
            "path": urdf_path,
            "pos": np.array(basePosition, dtype=float),
            "orn": np.array(baseOrientation, dtype=float),
            "vel": np.zeros(3, dtype=float),
            "omega": np.zeros(3, dtype=float),
            "joints": {0: 0.0, 1: 0.0}
        }
        if PYBULLET_AVAILABLE:
            try:
                real_id = real_p.loadURDF(urdf_path, basePosition, baseOrientation)
                # Keep tracking in self.bodies just in case
                self.bodies[real_id] = self.bodies.pop(body_id)
                return real_id
            except Exception:
                pass
        return body_id
        
    def resetBasePositionAndOrientation(self, body_id, position, orientation):
        if len(orientation) != 4:
            raise ValueError("Orientation must have exactly 4 elements.")
        if body_id in self.bodies:
            self.bodies[body_id]["pos"] = np.array(position, dtype=float)
            self.bodies[body_id]["orn"] = np.array(orientation, dtype=float)
        if PYBULLET_AVAILABLE:
            try:
                real_p.resetBasePositionAndOrientation(body_id, position, orientation)
            except Exception:
                pass
            
    def getBasePositionAndOrientation(self, body_id):
        if PYBULLET_AVAILABLE:
            try:
                return real_p.getBasePositionAndOrientation(body_id)
            except Exception:
                pass
        if body_id in self.bodies:
            b = self.bodies[body_id]
            return b["pos"].tolist(), b["orn"].tolist()
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]
        
    def setJointMotorControl2(self, bodyUniqueId, jointIndex, controlMode, targetVelocity=None, targetPosition=None, force=None, **kwargs):
        if PYBULLET_AVAILABLE:
            try:
                real_p.setJointMotorControl2(
                    bodyUniqueId, jointIndex, controlMode,
                    targetVelocity=targetVelocity, targetPosition=targetPosition,
                    force=force, **kwargs
                )
            except Exception:
                pass
        if bodyUniqueId in self.bodies:
            b = self.bodies[bodyUniqueId]
            if targetVelocity is not None:
                b["joints"][jointIndex] = targetVelocity
            elif targetPosition is not None:
                b["joints"][jointIndex] = targetPosition
            
            l_vel = b["joints"].get(0, 0.0)
            r_vel = b["joints"].get(1, 0.0)
            forward_speed = 0.5 * (l_vel + r_vel)
            yaw_rate = 0.5 * (r_vel - l_vel)
            
            b["vel"][0] = forward_speed
            b["omega"][2] = yaw_rate
            
    def stepSimulation(self):
        stepped_real = False
        if PYBULLET_AVAILABLE:
            try:
                real_p.stepSimulation()
                stepped_real = True
            except Exception:
                pass
        if stepped_real:
            return
        for b in self.bodies.values():
            z = b["orn"][2]
            w = b["orn"][3]
            yaw = 2.0 * np.arctan2(z, w)
            
            yaw += b["omega"][2] * self.time_step
            new_z = np.sin(yaw / 2.0)
            new_w = np.cos(yaw / 2.0)
            b["orn"] = np.array([0.0, 0.0, new_z, new_w], dtype=float)
            
            dx = b["vel"][0] * np.cos(yaw)
            dy = b["vel"][0] * np.sin(yaw)
            b["pos"][0] += dx * self.time_step
            b["pos"][1] += dy * self.time_step
            b["pos"][2] += b["vel"][2] * self.time_step
            
    def getKeyboardEvents(self):
        if PYBULLET_AVAILABLE:
            try:
                return real_p.getKeyboardEvents()
            except Exception:
                pass
        return {}
        
    def getLinkState(self, bodyUniqueId, linkIndex):
        if PYBULLET_AVAILABLE:
            try:
                return real_p.getLinkState(bodyUniqueId, linkIndex)
            except Exception:
                pass
        if bodyUniqueId in self.bodies:
            b = self.bodies[bodyUniqueId]
            return [b["pos"].tolist(), b["orn"].tolist()]
        return [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]

# Register pybullet in sys.modules if not present
if "pybullet" not in sys.modules:
    sys.modules["pybullet"] = MockPyBullet()
