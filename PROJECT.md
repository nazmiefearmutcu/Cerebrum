# Project: Cerebrum Architectural Remediation

## Architecture
- **Pillar 1 PC substrate:** error neurons $\epsilon_l = x_l - \hat{y}_l$ with PyTorch device-agnostic backend.
- **Pillar 2 local plasticity:** four-factor Hebbian with Kolen-Pollack alignment between $W$ and $B$.
- **Pillar 3 structured prior:** extended `GridHead` with support for non-commutative Lie group transformations (SO(3)/Heisenberg) for relational graphs (directed trees/hierarchies).
- **Pillar 4 stochastic inference:** Langevin noise settling with dynamic temperature / homeostasis under `MetaplasticFuse`.
- **Pillar 5 neuromorphic:** System 1 (reflex) bypass and System 2 (workspace settling) smooth transition; synchronized ROS node thread-safety.

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | M1: Subspace Segregation | Implement subspace segregation in `PCAreas` and `unified.py` | None | PLANNED |
| 2 | M2: Non-Commutative Prior | Support non-commutative Lie group rotations in `GridHead` for `TreeRelationalGraph` | None | PLANNED |
| 3 | M3: Metaplastic Homeostasis | Implement dynamic homeostasis and temperature adaptation in `MetaplasticFuse` | None | PLANNED |
| 4 | M4: System Coordination & KP | Smooth System 1-2 transitions, fix ROS thread safety, implement Kolen-Pollack and Gumbel-Max stability | None | PLANNED |
| 5 | M5: Verification & Bundle | Rebuild `cerebrum_submission.py`, run pytest suite, verify final forensic audit | M1, M2, M3, M4 | PLANNED |

## Interface Contracts
- **GridHead**:
  - `GridHead.step(motor_action)`: updates Lie group rotations.
- **MetaplasticFuse**:
  - `MetaplasticFuse.step(pi, eps, eligibility)`: local metaplastic update with homeostasis.
- **ROSNode**:
  - Expose safe thread-safe motor writing and speed reading.

## Code Layout
- `cerebrum/`: Core source files of the CerebrumNet model.
- `cerebrum_submission.py`: Bundled single-file submission.
- `tests/`: Test suite files.

## Hardware Requirements & Sim2Real Calibration
- **On-board Compute:** NVIDIA Jetson Orin Nano (8GB) or Raspberry Pi 5.
- **Communication Protocol:** CANopen / Modbus RTU interface (500 kbps) for motor control.
- **Pin Configuration (Typical GPIO mapping):**
  - GPIO 17 (Pin 11): CAN Tx
  - GPIO 18 (Pin 12): CAN Rx
  - GPIO 27, 22 (Pins 13, 15): Quadrature Encoder inputs A and B
- **Electrical Tolerances:**
  - Logic voltage: 3.3V (3.0V - 3.6V safe range)
  - Motor supply voltage: 12.0V - 24.0V nominal (Low-voltage cutoff at 11.1V)
  - Maximum current limits: 10.0 A per motor coil (soft-clamped to prevent winding burn).

## Simulation Setup & CLI Parameter Guide
- **Calibrate metrics database:**
  ```bash
  python metrics_collector.py --calibrate
  ```
- **Measure baseline power draw:**
  ```bash
  python power_parser.py --baseline
  ```
- **Run tray-balancing dynamics simulation (with 20% noise and motor command clamping):**
  ```bash
  python run_validation_sim.py --model cerebrum --episodes 500 --noise_level 0.20 --clamp_motor
  ```
- **Run continuous memory stress profiling:**
  ```bash
  pytest -s tests/test_stress.py
  ```
- **Run Counterfactual / Adversarial validation suites:**
  ```bash
  pytest tests/test_adversarial.py
  ```

