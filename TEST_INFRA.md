# E2E Test Infra: Cerebrum PyTorch, Grounding & Reflex Bypass

## Test Philosophy
- **Opaque-box, requirement-driven**: Tests derive directly from the requirements in `ORIGINAL_REQUEST.md` (latest follow-up) rather than implementation-specific internals.
- **Systematic methodology**: Test coverage is organized using the 4-tier testing hierarchy (Feature Coverage, Boundary/Corner Cases, Cross-Feature combinations, and Real-World Application scenarios).
- **Fallback resilience**: In environments where physical dependencies (like CUDA, MPS, or a full ROS 2 installation) are missing, tests fall back to mock interfaces and CPU execution gracefully.

## Feature Inventory
| # | Feature | Source (requirement) | Tier 1 | Tier 2 | Tier 3 |
|---|---------|---------------------|:------:|:------:|:------:|
| 1 | PyTorch Acceleration Backend (F1) | ORIGINAL_REQUEST R1 | 5      | 5      | ✓      |
| 2 | Sensory-Motor Grounding & PyBullet/ROS 2 (F2) | ORIGINAL_REQUEST R2 | 5      | 5      | ✓      |
| 3 | System 1 (Cerebellum) Reflex Bypass (F3) | ORIGINAL_REQUEST R3 | 5      | 5      | ✓      |

## Test Architecture
- **Test Runner**: Located at `tests/run_e2e_tests.py` and invokable via `python3 tests/run_e2e_tests.py` or standard `pytest tests/test_e2e.py`.
- **Interface compatibility**: Focuses on verification via public entry points:
  - `CerebrumNet` config options (`use_torch`, `device`).
  - Grounded agent loops, sensory/motor processor functions.
  - ROS 2 topics `/sensory_input` and `/motor_commands` published/subscribed.
  - System 1 reflex triggers and latency measurements.

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Complexity |
|---|----------|--------------------|------------|
| 1 | `test_scenario_room_navigation_pytorch` | F1, F2 | Medium |
| 2 | `test_scenario_obstacle_run_reflex` | F2, F3 | High |
| 3 | `test_scenario_device_agnostic_training` | F1, F2 | Medium |
| 4 | `test_scenario_ros2_control_loop` | F1, F2, F3 | High |
| 5 | `test_scenario_edge_recovery_and_sorting` | F1, F2, F3 | High |

## Coverage Thresholds
- **Tier 1 (Feature Coverage)**: 15 test cases (5 per feature).
- **Tier 2 (Boundary & Corner Cases)**: 15 test cases (5 per feature).
- **Tier 3 (Cross-Feature Combinations)**: 4 test cases covering major interactions.
- **Tier 4 (Real-World Application Scenarios)**: 5 test cases simulating complete tasks.
- **Total E2E test cases**: 39 test cases (exceeds the 38 minimum threshold).
