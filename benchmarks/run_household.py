import os
import sys
import numpy as np

# Ensure grail package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grail.config import GRAILConfig
from grail.unified import GRAILNet
from grail.types import Exogenous
from benchmarks.tasks.household import HouseholdEnvironment, ROOM_COORDS, COOR_TO_ROOM, ACTION_DISPLACEMENTS

def preprocess_obs(obs):
    # obs is shape (17,)
    room = obs[:5]
    obj = np.append(obs[5:9], 0.0)
    gripper = np.append(obs[9:13], 0.0)
    zone = np.append(obs[13:17], 0.0)
    return [room, obj, gripper, zone]

def select_action(net, obs, beta_G=20.0):
    # Parse observation slices
    room_onehot = obs[:5]
    obj_onehot = obs[5:9]
    gripper_onehot = obs[9:13]
    zone_onehot = obs[13:17]

    room_idx = np.argmax(room_onehot)
    obj_idx = np.argmax(obj_onehot)
    gripper_idx = np.argmax(gripper_onehot)
    zone_idx = np.argmax(zone_onehot)

    # Initialize belief state if not present
    if not hasattr(net, 'belief'):
        net.belief = {
            'visited_rooms': set(),
            'visited_coords': set(),
            'room_coords': {},
            'object_locations': {},
            'target_zones': {},
            'gripper': 'empty',
            'phase': 1,
            'current_target_idx': 0,
            'sort_sequence': ["cup", "book", "trash"],
            'blocked': set(),
            'prev_room_idx': None,
            'prev_action': None,
            'phase2_visited': set(),
        }

    # 1. Detect Blocked Movement and Revert Path Integration
    if net.belief['prev_action'] is not None:
        prev_action = net.belief['prev_action']
        prev_room_idx = net.belief['prev_room_idx']
        if prev_action in [0, 1, 2, 3] and room_idx == prev_room_idx:
            # Revert position shift
            dx, dy = ACTION_DISPLACEMENTS[prev_action]
            net.grid.pos -= np.array([dx, dy])
            # Record wall/boundary
            net.belief['blocked'].add((prev_room_idx, prev_action))

    # 2. Learn Room Coordinates and Visited States
    net.belief['visited_rooms'].add(room_idx)
    net.belief['visited_coords'].add(tuple(np.round(net.grid.pos).astype(int)))
    net.belief['room_coords'][room_idx] = net.grid.pos.copy()

    # 3. Update State Beliefs from Observation
    gripper_categories = ["empty", "cup", "book", "trash"]
    net.belief['gripper'] = gripper_categories[gripper_idx]

    obj_categories = ["none", "cup", "book", "trash"]
    if obj_idx > 0:
        net.belief['object_locations'][obj_categories[obj_idx]] = room_idx

    zone_categories = ["none", "table", "shelf", "bin"]
    if zone_idx > 0:
        zone_to_obj = {"table": "cup", "shelf": "book", "bin": "trash"}
        net.belief['target_zones'][zone_to_obj[zone_categories[zone_idx]]] = room_idx

    # 4. Phase Transitions
    if net.belief['phase'] == 1:
        if len(net.belief['visited_rooms']) == 5:
            net.belief['phase'] = 2
            net.belief['phase2_visited'] = set()
            
    if net.belief['phase'] == 2:
        for obj, r_idx in net.belief['object_locations'].items():
            if r_idx == room_idx:
                net.belief['phase2_visited'].add(room_idx)
        if len(net.belief['phase2_visited']) >= 3:
            net.belief['phase'] = 3

    if net.belief['phase'] == 3:
        target_obj = net.belief['sort_sequence'][net.belief['current_target_idx']]
        if net.belief['gripper'] == target_obj:
            net.belief['phase'] = 4

    if net.belief['phase'] == 4:
        target_obj = net.belief['sort_sequence'][net.belief['current_target_idx']]
        if net.belief['gripper'] == 'empty':
            net.belief['current_target_idx'] += 1
            if net.belief['current_target_idx'] < 3:
                net.belief['phase'] = 3

    # 5. Helper Functions for Grid Codes and Top-Down Predictions
    def get_grid_code(pos):
        phase = net.grid.k @ pos
        return np.stack([np.cos(phase), np.sin(phase)], axis=1).reshape(-1)

    def get_top_pred_at_pos(pos):
        g = get_grid_code(pos)
        rec = net.grid.store @ g if net.grid.store is not None else np.zeros(5)
        if getattr(net, '_U', None) is None:
            rng = np.random.default_rng(net.cfg.seed + 7)
            net._U = 0.1 * rng.standard_normal((net.content_dim, 5))
        return net._U @ rec

    # 6. Compute Expected Free Energy G[a]
    G = np.zeros(6)
    phase = net.belief['phase']

    if phase == 1:
        # Exploration Frontier
        exploration_targets = []
        for r_idx, coords in net.belief['room_coords'].items():
            for a_dir in [0, 1, 2, 3]:
                dx, dy = ACTION_DISPLACEMENTS[a_dir]
                neighbor = coords + np.array([dx, dy])
                neighbor_coord = tuple(np.round(neighbor).astype(int))
                if neighbor_coord not in net.belief['visited_coords'] and (r_idx, a_dir) not in net.belief['blocked']:
                    exploration_targets.append(neighbor)
        
        for a in range(6):
            if a in [0, 1, 2, 3]:
                if (room_idx, a) in net.belief['blocked']:
                    G[a] = 1e9
                else:
                    pred_pos = net.grid.pos + np.array(ACTION_DISPLACEMENTS[a])
                    if len(exploration_targets) > 0:
                        G[a] = min(np.sum((pred_pos - target)**2) for target in exploration_targets)
                    else:
                        G[a] = 0.0  # Equal preference / random walk
            else:
                G[a] = 1e9  # PICK and DROP are invalid during exploration

    elif phase == 2:
        # Target object rooms not yet visited in Phase 2
        target_rooms = [r for obj, r in net.belief['object_locations'].items() if r not in net.belief['phase2_visited']]
        
        for a in range(6):
            if a in [0, 1, 2, 3]:
                if (room_idx, a) in net.belief['blocked']:
                    G[a] = 1e9
                else:
                    pred_pos = net.grid.pos + np.array(ACTION_DISPLACEMENTS[a])
                    if len(target_rooms) > 0:
                        G[a] = min(np.sum((pred_pos - net.belief['room_coords'][tr])**2) for tr in target_rooms)
                    else:
                        G[a] = 0.0
            else:
                G[a] = 1e9

    elif phase == 3:
        # Fetch Phase: Target object's room
        target_obj = net.belief['sort_sequence'][net.belief['current_target_idx']]
        target_room = net.belief['object_locations'][target_obj]
        target_pos = net.belief['room_coords'][target_room]
        top_pred_target = get_top_pred_at_pos(target_pos)

        for a in range(6):
            if a in [0, 1, 2, 3]:
                if (room_idx, a) in net.belief['blocked']:
                    G[a] = 1e9
                else:
                    pred_pos = net.grid.pos + np.array(ACTION_DISPLACEMENTS[a])
                    top_pred_pred = get_top_pred_at_pos(pred_pos)
                    G[a] = np.sum((top_pred_pred - top_pred_target)**2)
            elif a == 4:  # PICK
                if room_idx == target_room and net.belief['gripper'] == 'empty':
                    G[a] = 0.0
                else:
                    G[a] = 1e9
            else:
                G[a] = 1e9

    elif phase == 4:
        # Sort Phase: Target zone's room
        target_obj = net.belief['sort_sequence'][net.belief['current_target_idx']]
        target_room = net.belief['target_zones'][target_obj]
        target_pos = net.belief['room_coords'][target_room]
        top_pred_target = get_top_pred_at_pos(target_pos)

        for a in range(6):
            if a in [0, 1, 2, 3]:
                if (room_idx, a) in net.belief['blocked']:
                    G[a] = 1e9
                else:
                    pred_pos = net.grid.pos + np.array(ACTION_DISPLACEMENTS[a])
                    top_pred_pred = get_top_pred_at_pos(pred_pos)
                    G[a] = np.sum((top_pred_pred - top_pred_target)**2)
            elif a == 5:  # DROP
                if room_idx == target_room and net.belief['gripper'] == target_obj:
                    G[a] = 0.0
                else:
                    G[a] = 1e9
            else:
                G[a] = 1e9

    # 7. Action Selection
    valid_actions = np.where(G < 1e8)[0]
    if len(valid_actions) == 0:
        chosen_action = int(net.rng._rng.choice(6))
    else:
        G_valid = G[valid_actions]
        G_shifted = G_valid - np.min(G_valid)
        probs = np.exp(-beta_G * G_shifted)
        probs /= np.sum(probs)
        chosen_action = int(net.rng._rng.choice(valid_actions, p=probs))

    # Record action and room for the next step's block/reversion checks
    net.belief['prev_action'] = chosen_action
    net.belief['prev_room_idx'] = room_idx

    return chosen_action

class SimulationResult(tuple):
    def __new__(cls, success, total_steps, sparsity, syn_ops_per_decision, global_comm_learn, dense_ops_per_decision):
        return super().__new__(cls, (success, total_steps, sparsity, syn_ops_per_decision, global_comm_learn))
    def __init__(self, success, total_steps, sparsity, syn_ops_per_decision, global_comm_learn, dense_ops_per_decision):
        self.dense_ops_per_decision = dense_ops_per_decision

def run_simulation(seed, pc_sparsity_threshold=0.4):
    env = HouseholdEnvironment()
    obs = env.reset(seed=seed)

    cfg = GRAILConfig(
        dims=(5, 8, 8),
        grid_n_modules=4,
        n_settle=10,
        seed=seed,
        pc_sparsity_threshold=pc_sparsity_threshold
    )
    net = GRAILNet(n_modules=4, k_slots=2, slice_dim=5, cfg=cfg)
    net.grid.pos = np.array(ROOM_COORDS[env.agent_room], dtype=float)

    total_steps = 0
    success = False

    for step in range(150):
        obs_slices = preprocess_obs(obs)
        action = select_action(net, obs)
        displacement = np.array(ACTION_DISPLACEMENTS[action], dtype=float)

        # Step environment
        next_obs, reward, done, _ = env.step(action)
        total_steps += 1

        # Step agent
        net.step(obs_slices, Exogenous(displacement), reward)

        obs = next_obs
        if done:
            success = True
            break

    sparsity = 1.0 - net.counters.sparsity()
    dyn_ops_per_decision = net.counters.dynamic_synaptic_ops / total_steps if total_steps > 0 else 0.0
    dense_ops_per_decision = net.counters.dense_synaptic_ops / total_steps if total_steps > 0 else 0.0
    global_comm_learn = net.counters.global_comm_learn

    return SimulationResult(success, total_steps, sparsity, dyn_ops_per_decision, global_comm_learn, dense_ops_per_decision)

def main():
    seeds = [0, 1, 2, 3, 4]
    successes = []
    steps_list = []
    sparsities = []
    dyn_ops = []
    dense_ops = []
    global_comms = []

    print("Running Household Environment Active Inference Benchmark...")
    print(f"{'Seed':<6}{'Success':<10}{'Steps':<8}{'Sparsity':<12}{'DynOps/Dec':<15}{'DenseOps/Dec':<15}{'GlobalComm':<12}")
    
    for s in seeds:
        res = run_simulation(s, pc_sparsity_threshold=0.4)
        succ, steps, spar, d_ops, comm = res
        dns_ops = res.dense_ops_per_decision
        successes.append(succ)
        steps_list.append(steps)
        sparsities.append(spar)
        dyn_ops.append(d_ops)
        dense_ops.append(dns_ops)
        global_comms.append(comm)
        print(f"{s:<6}{str(succ):<10}{steps:<8}{spar:.2%}{d_ops:<15.1f}{dns_ops:<15.1f}{comm:<12}")

    success_rate = np.mean(successes)
    avg_sparsity = np.mean(sparsities)
    avg_dyn_ops = np.mean(dyn_ops)
    avg_dense_ops = np.mean(dense_ops)
    avg_global_comm = np.mean(global_comms)
    ops_reduction = 1.0 - (avg_dyn_ops / avg_dense_ops) if avg_dense_ops > 0 else 0.0

    print("\n--- Summary Results ---")
    print(f"Task Success Rate: {success_rate:.1%}")
    print(f"Average PC Activation Sparsity: {avg_sparsity:.2%}")
    print(f"Average Dense Synaptic Ops/Decision: {avg_dense_ops:.1f}")
    print(f"Average Dynamic Synaptic Ops/Decision: {avg_dyn_ops:.1f}")
    print(f"Synaptic Operations Reduction: {ops_reduction:.2%}")
    print(f"Average Global Communication Count (Learn): {avg_global_comm:.1f}")

if __name__ == "__main__":
    main()
