import numpy as np

# Coordinates of rooms
ROOM_COORDS = {
    "Living Room": (0, 0),
    "Kitchen": (0, 1),
    "Bedroom": (1, 0),
    "Bathroom": (1, 1),
    "Study": (0, -1),
}

COOR_TO_ROOM = {v: k for k, v in ROOM_COORDS.items()}
ROOM_NAMES = ["Living Room", "Kitchen", "Bedroom", "Bathroom", "Study"]

# Action displacements
ACTION_DISPLACEMENTS = {
    0: (0, 1),   # North
    1: (0, -1),  # South
    2: (1, 0),   # East
    3: (-1, 0),  # West
    4: (0, 0),   # PICK
    5: (0, 0),   # DROP
}

class HouseholdEnvironment:
    def __init__(self):
        self.rng = None
        self.rooms_list = ROOM_NAMES
        self.target_zones = {
            "cup": "Living Room",   # table is in Living Room
            "book": "Bedroom",       # shelf is in Bedroom
            "trash": "Kitchen",      # bin is in Kitchen
        }
        self.reset()

    def reset(self, seed=None):
        self.rng = np.random.default_rng(seed)
        
        # Randomize object starting rooms (distinct from their target zones, and distinct from each other)
        while True:
            chosen_rooms = self.rng.choice(self.rooms_list, size=3, replace=False)
            if (chosen_rooms[0] != "Living Room" and  # cup target
                chosen_rooms[1] != "Bedroom" and      # book target
                chosen_rooms[2] != "Kitchen"):        # trash target
                break
        
        self.object_rooms = {
            "cup": chosen_rooms[0],
            "book": chosen_rooms[1],
            "trash": chosen_rooms[2],
        }
        
        # Start at a random room
        self.agent_room = self.rng.choice(self.rooms_list)
        self.gripper = "empty"
        
        # Sequential phases: 1 (Navigation), 2 (Identification), 3 (Fetch), 4 (Sort)
        self.phase = 1
        self.visited_rooms = {self.agent_room}
        self.identified_objects = {obj: False for obj in ["cup", "book", "trash"]}
        
        self.sort_sequence = ["cup", "book", "trash"]
        self.current_target_index = 0
        
        return self.get_obs()

    def get_obs(self):
        # 5-dim Room ID
        room_idx = self.rooms_list.index(self.agent_room)
        room_onehot = np.zeros(5)
        room_onehot[room_idx] = 1.0
        
        # 4-dim Detected Object [none, cup, book, trash]
        detected = "none"
        for obj, room in self.object_rooms.items():
            if room == self.agent_room:
                detected = obj
                break
        obj_categories = ["none", "cup", "book", "trash"]
        obj_idx = obj_categories.index(detected)
        obj_onehot = np.zeros(4)
        obj_onehot[obj_idx] = 1.0
        
        # 4-dim Gripper State [empty, cup, book, trash]
        gripper_categories = ["empty", "cup", "book", "trash"]
        gripper_idx = gripper_categories.index(self.gripper)
        gripper_onehot = np.zeros(4)
        gripper_onehot[gripper_idx] = 1.0
        
        # 4-dim Target Zone [none, table, shelf, bin]
        zone_mapping = {
            "Living Room": "table",
            "Bedroom": "shelf",
            "Kitchen": "bin",
        }
        zone_in_room = zone_mapping.get(self.agent_room, "none")
        zone_categories = ["none", "table", "shelf", "bin"]
        zone_idx = zone_categories.index(zone_in_room)
        zone_onehot = np.zeros(4)
        zone_onehot[zone_idx] = 1.0
        
        obs = np.concatenate([room_onehot, obj_onehot, gripper_onehot, zone_onehot])
        return obs

    def step(self, action):
        reward = 0.0
        done = False
        
        # 0: North, 1: South, 2: East, 3: West
        if action in [0, 1, 2, 3]:
            curr_coords = ROOM_COORDS[self.agent_room]
            dx, dy = ACTION_DISPLACEMENTS[action]
            new_coords = (curr_coords[0] + dx, curr_coords[1] + dy)
            if new_coords in COOR_TO_ROOM:
                self.agent_room = COOR_TO_ROOM[new_coords]
            
            # Update Phase 1
            if self.phase == 1:
                if self.agent_room not in self.visited_rooms:
                    self.visited_rooms.add(self.agent_room)
                    reward += 1.0
                if len(self.visited_rooms) == 5:
                    self.phase = 2
                    # Transition to Phase 2: identify any object in current room immediately
                    for obj, room in self.object_rooms.items():
                        if not self.identified_objects[obj] and self.agent_room == room:
                            self.identified_objects[obj] = True
                            reward += 1.0
                    if all(self.identified_objects.values()):
                        self.phase = 3
            # Update Phase 2
            elif self.phase == 2:
                for obj, room in self.object_rooms.items():
                    if not self.identified_objects[obj] and self.agent_room == room:
                        self.identified_objects[obj] = True
                        reward += 1.0
                if all(self.identified_objects.values()):
                    self.phase = 3
                    
        elif action == 4:  # PICK
            if self.phase == 3:
                target_obj = self.sort_sequence[self.current_target_index]
                if self.agent_room == self.object_rooms[target_obj] and self.gripper == "empty":
                    self.gripper = target_obj
                    self.object_rooms[target_obj] = "gripper"
                    reward += 1.0
                    self.phase = 4
                    
        elif action == 5:  # DROP
            if self.phase == 4:
                target_obj = self.sort_sequence[self.current_target_index]
                target_zone_room = self.target_zones[target_obj]
                if self.gripper == target_obj and self.agent_room == target_zone_room:
                    self.gripper = "empty"
                    self.object_rooms[target_obj] = target_zone_room
                    reward += 1.0
                    self.current_target_index += 1
                    if self.current_target_index == 3:
                        done = True
                    else:
                        self.phase = 3
                        
        obs = self.get_obs()
        return obs, reward, done, {}
