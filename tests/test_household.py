import numpy as np
import pytest

from benchmarks.tasks.household import HouseholdEnvironment, ROOM_COORDS, COOR_TO_ROOM, ACTION_DISPLACEMENTS, ROOM_NAMES
from benchmarks.run_household import run_simulation, select_action, preprocess_obs
from cerebrum.config import CerebrumConfig
from cerebrum.unified import CerebrumNet
from cerebrum.types import Exogenous

def test_room_transitions():
    """Verify room coordinates and transition boundaries (physical walls)."""
    env = HouseholdEnvironment()
    env.reset(seed=42)
    
    # Force agent starting position to Living Room
    env.agent_room = "Living Room"
    
    # 1. Living Room (0,0) -> North (0,1) is Kitchen
    obs, reward, done, _ = env.step(0)  # North
    assert env.agent_room == "Kitchen"
    
    # 2. Kitchen (0,1) -> North (0,2) is invalid, should stay in Kitchen
    obs, reward, done, _ = env.step(0)  # North
    assert env.agent_room == "Kitchen"
    
    # 3. Kitchen (0,1) -> East (1,1) is Bathroom
    obs, reward, done, _ = env.step(2)  # East
    assert env.agent_room == "Bathroom"
    
    # 4. Bathroom (1,1) -> South (1,0) is Bedroom
    obs, reward, done, _ = env.step(1)  # South
    assert env.agent_room == "Bedroom"
    
    # 5. Bedroom (1,0) -> West (0,0) is Living Room
    obs, reward, done, _ = env.step(3)  # West
    assert env.agent_room == "Living Room"
    
    # 6. Living Room (0,0) -> South (0,-1) is Study
    obs, reward, done, _ = env.step(1)  # South
    assert env.agent_room == "Study"
    
    # 7. Study (0,-1) -> South (0,-2) is invalid, stays in Study
    obs, reward, done, _ = env.step(1)  # South
    assert env.agent_room == "Study"

def test_phase_progression_and_rewards():
    """Verify the 4-phase sequential state machine and correct reward collection."""
    env = HouseholdEnvironment()
    env.reset(seed=42)
    
    # Initialize objects at fixed rooms that are NOT target zones
    env.object_rooms = {
        "cup": "Study",      # target zone is Living Room
        "book": "Kitchen",    # target zone is Bedroom
        "trash": "Bathroom",  # target zone is Kitchen
    }
    env.agent_room = "Living Room"
    env.phase = 1
    env.visited_rooms = {"Living Room"}
    
    # Phase 1: Navigation
    # Visit Kitchen (0,1)
    obs, reward, done, _ = env.step(0) # North
    assert env.agent_room == "Kitchen"
    assert reward == 1.0
    assert env.phase == 1
    
    # Visit Bathroom (1,1)
    obs, reward, done, _ = env.step(2) # East
    assert env.agent_room == "Bathroom"
    assert reward == 1.0
    assert env.phase == 1
    
    # Visit Bedroom (1,0)
    obs, reward, done, _ = env.step(1) # South
    assert env.agent_room == "Bedroom"
    assert reward == 1.0
    assert env.phase == 1
    
    # Visit Living Room (already visited)
    obs, reward, done, _ = env.step(3) # West
    assert env.agent_room == "Living Room"
    assert reward == 0.0
    assert env.phase == 1
    
    # Visit Study (0,-1) to complete Phase 1
    # Study has "cup". When Study is visited:
    # Phase 1 completes (5 rooms visited).
    # Transition to Phase 2.
    # In Phase 2, we immediately identify "cup" (since agent is in Study).
    # Thus, total reward should be:
    #   +1.0 (for visiting Study, new room)
    #   +1.0 (for identifying cup, which is in Study)
    # Since "book" (Kitchen) and "trash" (Bathroom) are in other rooms, they are not identified yet.
    # So Phase 2 is NOT complete.
    obs, reward, done, _ = env.step(1) # South
    assert env.agent_room == "Study"
    assert reward == 2.0  # 1.0 (new room) + 1.0 (identify cup)
    assert env.phase == 2
    assert env.identified_objects["cup"] is True
    assert env.identified_objects["book"] is False
    assert env.identified_objects["trash"] is False
    
    # Phase 2: Identification
    # Visit Living Room (no object)
    obs, reward, done, _ = env.step(0) # North
    assert reward == 0.0
    assert env.phase == 2
    
    # Visit Kitchen (contains book)
    obs, reward, done, _ = env.step(0) # North (goes to Kitchen)
    assert env.agent_room == "Kitchen"
    assert reward == 1.0  # +1.0 for identifying book
    assert env.identified_objects["book"] is True
    assert env.phase == 2
    
    # Visit Bathroom (contains trash) -> completes Phase 2 -> transitions to Phase 3
    obs, reward, done, _ = env.step(2) # East (goes to Bathroom)
    assert env.agent_room == "Bathroom"
    assert reward == 1.0  # +1.0 for identifying trash
    assert env.identified_objects["trash"] is True
    assert env.phase == 3

def test_fetch_and_sort():
    """Verify PICK and DROP rules, gripper states, and phase transitions."""
    env = HouseholdEnvironment()
    env.reset(seed=42)
    
    # Manually configure the state to Phase 3 (Fetch)
    env.phase = 3
    env.object_rooms = {
        "cup": "Study",
        "book": "Kitchen",
        "trash": "Bathroom",
    }
    env.agent_room = "Living Room"
    env.gripper = "empty"
    env.current_target_index = 0  # target: cup
    
    # Invalid PICK: wrong room (no cup in Living Room)
    obs, reward, done, _ = env.step(4) # PICK
    assert reward == 0.0
    assert env.gripper == "empty"
    assert env.phase == 3
    
    # Navigate to Study (contains cup)
    env.step(1)  # South
    assert env.agent_room == "Study"
    
    # Valid PICK: cup picked up
    obs, reward, done, _ = env.step(4) # PICK
    assert reward == 1.0
    assert env.gripper == "cup"
    assert env.object_rooms["cup"] == "gripper"
    assert env.phase == 4  # Transition to Sort
    
    # Invalid DROP: wrong room (target zone for cup is Living Room, currently in Study)
    obs, reward, done, _ = env.step(5) # DROP
    assert reward == 0.0
    assert env.gripper == "cup"
    assert env.phase == 4
    
    # Navigate to Living Room (0,0)
    env.step(0)  # North
    assert env.agent_room == "Living Room"
    
    # Valid DROP: cup dropped at table (Living Room)
    obs, reward, done, _ = env.step(5) # DROP
    assert reward == 1.0
    assert env.gripper == "empty"
    assert env.object_rooms["cup"] == "Living Room"
    assert env.current_target_index == 1  # Next target: book
    assert env.phase == 3  # Return to Fetch

def test_active_inference_action_selection():
    """Verify stochastically minimizing EFE G(a) using grid code distances."""
    env = HouseholdEnvironment()
    env.reset(seed=42)
    
    # Force agent position and grid head alignment
    env.agent_room = "Living Room"
    obs = env.get_obs()
    
    cfg = CerebrumConfig(dims=(5, 8, 8), grid_n_modules=4, seed=42)
    net = CerebrumNet(n_modules=4, k_slots=2, slice_dim=5, cfg=cfg)
    net.grid.pos = np.array(ROOM_COORDS["Living Room"], dtype=float)
    
    # Initialize the belief state to match the test conditions
    # (Agent is in Living Room, but belief reflects that Living Room, Bedroom,
    # Bathroom, and Study have been visited. Only Kitchen is unvisited).
    net.belief = {
        'visited_rooms': {0, 2, 3, 4},  # Living Room, Bedroom, Bathroom, Study
        'visited_coords': {(0, 0), (1, 0), (1, 1), (0, -1)},
        'room_coords': {
            0: np.array(ROOM_COORDS["Living Room"], dtype=float),
            2: np.array(ROOM_COORDS["Bedroom"], dtype=float),
            3: np.array(ROOM_COORDS["Bathroom"], dtype=float),
            4: np.array(ROOM_COORDS["Study"], dtype=float),
        },
        'object_locations': {},
        'target_zones': {},
        'gripper': 'empty',
        'phase': 1,
        'current_target_idx': 0,
        'sort_sequence': ["cup", "book", "trash"],
        'blocked': {(0, 3)},  # Blocked West from Living Room
        'prev_room_idx': None,
        'prev_action': None,
    }
    
    # Step grid to bind the current coordinate so that get_top_pred works
    obs_slices = preprocess_obs(obs)
    obs_mean = np.mean(np.stack([np.asarray(o, float) for o in obs_slices]), axis=0)
    net.grid.bind(obs_mean, M=1.0)
    
    # Run action selection (should select North [0] deterministically)
    action = select_action(net, obs, beta_G=100.0)
    assert action == 0

def test_neuromorphic_metrics_and_sparsity():
    """Verify that E2E simulation successfully completes and sparsity >= 80%."""
    # Run the benchmark simulation for seed 0
    success, total_steps, sparsity, syn_ops_per_decision, global_comm_learn = run_simulation(0, pc_sparsity_threshold=0.4)
    
    assert success is True
    assert total_steps <= 150
    assert sparsity >= 0.80
    assert syn_ops_per_decision > 0
    # Global learn communication count must be O(1) per step, meaning it scales with steps
    assert global_comm_learn == total_steps
