import os
import json

# Predefined robot configs
STANDARD_ROBOTS = {
    "Unitree G1": {
        "name": "Unitree G1",
        "height": "1.27 m",
        "weight": "35 kg",
        "dof": 23,
        "class": "Humanoid",
        "joints": [
            "neck_yaw", "neck_pitch",
            "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist",
            "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist",
            "waist_yaw", "waist_pitch",
            "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle",
            "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle"
        ],
        "sensors": ["3D LiDAR", "Depth Camera", "IMU", "Joint Encoders"],
        "macros": {
            "clean_house": {
                "name": "Clean the House",
                "description": "Explores the household environment, detects scattered clutter (cups, books, trash), retrieves them, and sorts them into their target destinations (living room, bedroom, kitchen). Uses Hebbian learning and path-integration grid maps to navigate.",
                "icon": "🧹",
                "phases": [
                    "Phase 1: Exploration & Mapping",
                    "Phase 2: Locating and Identifying Clutter",
                    "Phase 3: Grasping and Collecting Items",
                    "Phase 4: Delivering and Sorting into Target Zones",
                    "Completed: Task Succeeded"
                ]
            },
            "serve_coffee": {
                "name": "Serve Coffee",
                "description": "Retrieves a coffee mug from the cupboard, places it under the espresso dispenser, triggers brewing, and delivers the hot cup to the desk.",
                "icon": "☕",
                "phases": [
                    "Locating Mug Cupboard",
                    "Opening Cabinet and Grasping Mug",
                    "Placing Mug on Drip Tray",
                    "Initiating Espresso Brewing Cycle",
                    "Transporting Coffee to Desk",
                    "Completed: Coffee Served"
                ]
            },
            "patrol_area": {
                "name": "Patrol Area",
                "description": "Executes a safety patrol sweep around the house in a loop, utilizing 3D LiDAR to scan for structural changes or obstacle blockages.",
                "icon": "🛡️",
                "phases": [
                    "Initializing LiDAR Systems",
                    "Scanning Living Room Perimeter",
                    "Sweeping Kitchen & Corridor",
                    "Checking Study for Obstacles",
                    "Returning to Charge Station",
                    "Completed: Area Secure"
                ]
            }
        }
    },
    "Optimus Gen 2": {
        "name": "Optimus Gen 2",
        "height": "1.73 m",
        "weight": "56 kg",
        "dof": 39,
        "class": "Humanoid",
        "joints": [
            "neck_yaw", "neck_pitch", "neck_roll",
            "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
            "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
            "torso_yaw", "torso_pitch",
            "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
            "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
            "finger_joint_1", "finger_joint_2", "finger_joint_3", "finger_joint_4", "finger_joint_5"
        ],
        "sensors": ["Autopilot Cameras", "Tactile Finger Sensors", "IMU", "Joint Force Sensors"],
        "macros": {
            "fold_laundry": {
                "name": "Fold Laundry",
                "description": "Sorts and folds laundry items (shirts, pants) using fine-motor control, placing them in neat piles.",
                "icon": "👕",
                "phases": [
                    "Retrieving Shirt from Basket",
                    "Aligning Sleeves and Shoulders",
                    "Executing Folding Sequence",
                    "Stacking Folded Garment on Shelf",
                    "Completed: Laundry Folded"
                ]
            },
            "organize_tools": {
                "name": "Organize Tools",
                "description": "Identifies scattered items (wrenches, screwdrivers) on a workbench, sorts them by category, and files them in a drawer.",
                "icon": "🔧",
                "phases": [
                    "Scanning Workbench for Tools",
                    "Classifying Screwdrivers & Wrenches",
                    "Opening Tool Drawer",
                    "Placing Tools in Organized Layout",
                    "Completed: Workspace Organized"
                ]
            }
        }
    },
    "BD Atlas": {
        "name": "BD Atlas",
        "height": "1.50 m",
        "weight": "89 kg",
        "dof": 28,
        "class": "Humanoid",
        "joints": [
            "neck_yaw", "neck_pitch",
            "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow", "left_wrist",
            "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist",
            "torso_pitch", "torso_roll", "torso_yaw",
            "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle",
            "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle"
        ],
        "sensors": ["Multi-Spectral LiDAR", "Depth Sensors", "Joint Force Sensors"],
        "macros": {
            "carry_heavy_box": {
                "name": "Carry Heavy Box",
                "description": "Sinks center of gravity, grasps a heavy cargo bin with high grip torque, and carries it across unlevel terrain.",
                "icon": "📦",
                "phases": [
                    "Squatting and Aligning with Box",
                    "Lifting Cargo to Chest Level",
                    "Carrying Across Obstacle Field",
                    "Aligning and Placing on Platform",
                    "Completed: Box Deposited"
                ]
            },
            "backflip_demo": {
                "name": "Backflip Routine",
                "description": "Launches vertically into a backflip, tucking legs and executing a balanced landing with active torque absorption.",
                "icon": "🤸",
                "phases": [
                    "Preparing Launch Posture",
                    "Explosive Upward Jump",
                    "Mid-Air Rotation and Tuck",
                    "Spreading Legs for Contact",
                    "Damping Landing Force & Standing Tall",
                    "Completed: Backflip Perfected"
                ]
            }
        }
    },
    "Figure 01": {
        "name": "Figure 01",
        "height": "1.60 m",
        "weight": "60 kg",
        "dof": 20,
        "class": "Humanoid",
        "joints": [
            "neck_yaw", "neck_pitch",
            "left_shoulder_pitch", "left_shoulder_roll", "left_elbow", "left_wrist",
            "right_shoulder_pitch", "right_shoulder_roll", "right_elbow", "right_wrist",
            "waist_yaw", "waist_pitch",
            "left_hip_pitch", "left_hip_roll", "left_knee", "left_ankle",
            "right_hip_pitch", "right_hip_roll", "right_knee", "right_ankle"
        ],
        "sensors": ["AI Vision System", "Microphone Array", "Joint Torque Encoders"],
        "macros": {
            "serve_dinner": {
                "name": "Serve Dinner",
                "description": "Picks up dinner trays or cups from a counter, navigates to the dining area, and sets them down smoothly while giving a verbal greeting.",
                "icon": "🍽️",
                "phases": [
                    "Grasping Dinner Plate Tray",
                    "Balancing Tray during Locomotion",
                    "Approaching Dining Table",
                    "Placing Plates with Low Torque Impact",
                    "Issuing Verbal Greeting",
                    "Completed: Dinner Served"
                ]
            },
            "load_dishwasher": {
                "name": "Load Dishwasher",
                "description": "Locates dirty plates in the sink, rinses them under running water, opens the dishwasher door, and nests them in the racks.",
                "icon": "🧼",
                "phases": [
                    "Clearing Solid Waste from Plates",
                    "Water Rinse under Faucet",
                    "Opening Dishwasher Tray",
                    "Loading Plates into Lower Rack",
                    "Completed: Dishwasher Loaded"
                ]
            }
        }
    }
}

# Config path for custom robots
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
CUSTOM_CONFIG_FILE = os.path.join(CONFIG_DIR, "custom_robots.json")

def load_custom_robots():
    """Load user-defined custom robots from disk."""
    if not os.path.exists(CUSTOM_CONFIG_FILE):
        return {}
    try:
        with open(CUSTOM_CONFIG_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading custom robots: {e}")
        return {}

def save_custom_robot(robot_config):
    """Save a user-defined custom robot configuration."""
    custom = load_custom_robots()
    name = robot_config.get("name")
    if not name:
        raise ValueError("Robot config must have a unique name.")
    
    # Fill in defaults if not provided
    robot_config["class"] = robot_config.get("class", "Custom")
    robot_config["height"] = robot_config.get("height", "N/A")
    robot_config["weight"] = robot_config.get("weight", "N/A")
    robot_config["dof"] = len(robot_config.get("joints", []))
    
    # Add dummy macros for custom robots so they have actions
    robot_config["macros"] = robot_config.get("macros", {
        "custom_task": {
            "name": "Perform Custom Routine",
            "description": "Execute a pre-configured cognitive routine using Hebbian associations and joint coordination on the custom-configured robot.",
            "icon": "🤖",
            "phases": [
                "Calibrating Custom Sensors",
                "Synchronizing Joint Actuators",
                "Navigating to Target Space",
                "Executing Arm/Base Movement",
                "Verifying Output Accuracy",
                "Completed: Routine Finished"
            ]
        }
    })
    
    custom[name] = robot_config
    
    try:
        with open(CUSTOM_CONFIG_FILE, "w") as f:
            json.dump(custom, f, indent=4)
        return True
    except Exception as e:
        print(f"Error saving custom robot: {e}")
        return False

def get_all_robots():
    """Return dictionary combining standard and custom robots."""
    all_robots = STANDARD_ROBOTS.copy()
    custom_robots = load_custom_robots()
    all_robots.update(custom_robots)
    return all_robots
