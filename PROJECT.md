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
