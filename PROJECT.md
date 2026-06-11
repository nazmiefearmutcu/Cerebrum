# Project: Cerebrum Household Active Inference Agent

## Architecture
- **Environment** (`benchmarks/tasks/household.py`):
  - A graph or grid-world of connected rooms (e.g., Living Room, Kitchen, Bedroom, Bathroom, Study).
  - Pre-processed object identifiers (e.g., cup, book, trash) and drop-off zones (table, shelf, bin).
  - Multi-stage sequence: Navigation (room layout mapping), Object Identification, Fetch/Manipulate, Sorting/Cleaning.
- **Cerebrum Agent**:
  - Closed-loop active inference controller using `CerebrumNet`.
  - Sensory vectors: low-dimensional concatenation/mapping of object and room identifiers.
  - Active Inference action selection: evaluate potential actions, project next states, minimize predictive error and free energy $F$.
  - Grid prior path integration driven by motor efference copy.
- **Neuromorphic Metrics**:
  - Average activation sparsity $\rho \ge 80\%$ (active fraction $\le 20\%$) in PC areas.
  - Synaptic operations per decision logged.
  - Learn-time global communication restricted to scalar neuromodulator $M$.

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | M1: Household Environment | Implement simulated household room graph, objects, and drop-off logic | None | PLANNED |
| 2 | M2: Active Inference Agent | Implement Cerebrum controller with action selection via free energy minimization | 1 | PLANNED |
| 3 | M3: Benchmarks & Metrics | Implement benchmarks/run_household.py, collect metrics (sparsity, ops, scalar-M) | 2 | PLANNED |
| 4 | M4: E2E Verification | Run full suite over 5 seeds, ensure success rate >= 80%, sparsity >= 80%, and run audit | 3 | PLANNED |

## Interface Contracts
- **Environment API**:
  - `env.reset(seed)`: Returns initial observation vector (room and object identifiers).
  - `env.step(action)`: Accepts action, returns `(obs, reward, done, info)`.
- **Agent API**:
  - `agent.select_action(obs)`: Selects action minimizing free energy.
  - `agent.update(obs, action, reward)`: Updates internal PC states, grid head, and plasticity weights.

## Code Layout
- `cerebrum/`: Core source files of the CerebrumNet model.
- `benchmarks/tasks/household.py`: Simulated environment.
- `benchmarks/run_household.py`: Benchmark script.
- `tests/test_household.py`: Unit and integration tests.
