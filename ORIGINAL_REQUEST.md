# Original User Request

## 2026-06-09T13:11:23Z

Resolve the competition between the factored latent and the frozen grid prior in the full CerebrumNet architecture (P0), so that the factorized representation survives integration.

Working directory: /Users/nazmi/cerebrum
Integrity mode: development

## Requirements

### R1. Diagnostic Investigation
Isolate and identify the specific coupling mechanism in the unified `CerebrumNet.step` (such as workspace broadcast during training, gate selection dynamics, or settle/learn ordering) that disrupts the cortical module's factorized latent representation.

### R2. Cooperation Fix Implementation
Design and implement a clean, architecture-compliant fix that allows the factorized representation to survive training and settle in the full `CerebrumNet` environment. The fix must be enabled via an opt-in configuration flag (default behavior must remain unchanged).

### R3. Invariant Safety
All changes must strictly preserve the existing non-negotiable invariants (no backprop/autograd, no weight transport, scalar-only neuromodulator `M`, exogenous-only action `z_act`, and strict one-hot workspace writes).

## Acceptance Criteria

### Verification
- [ ] Running `python3 -m pytest` passes completely.
- [ ] Running the factorization pipeline with grid precision balancing (`CEREBRUM_BALANCE_GRID_PRECISION=1 python3 benchmarks/run_factorization_pipeline.py`) reports that the `full` condition trained decode accuracy is >= 0.80.
- [ ] Task 1 few-shot graph completion benchmark (`python3 benchmarks/run_task1.py`) runs successfully and confirms that the grid prior's few-shot performance is not degraded by the fix.

## Follow-up — 2026-06-09T13:47:28Z

Build a non-metric/asymmetric relational task and benchmark CEREBRUM-grid, flat-prior, and backprop-MLP baselines to map where the metric grid prior's assumptions degrade.

Working directory: /Users/nazmi/cerebrum
Integrity mode: development

## Requirements

### R1. Non-Metric Relational Task Creation
Design and implement a relational task/dataset representing abstract, non-metric, or directed/asymmetric relations (e.g., a directed tree or hierarchy) where transitions do not compose as commuting grid-rotations.

### R2. Benchmarking and Comparison
Run few-shot evaluation (K=5, 10, 20 observations) and compare prediction accuracy of:
1. CEREBRUM with the Lie-group rotational grid prior.
2. CEREBRUM with a flat/identity prior (ablating grid structure).
3. A backprop-MLP/transformer baseline.

### R3. Scaling Frontier Mapping
Document the multi-seed CI results and honestly map where the grid prior degrades compared to baselines (FM7 probe).

## Acceptance Criteria

### Verification
- [ ] Task and benchmark scripts run successfully.
- [ ] Pytest suite passes fully including any new tests.
- [ ] Results (accuracy plots or tables with 95% CIs) are generated and documented in the README.

## Follow-up — 2026-06-09T14:25:25Z

Implement and benchmark a multimodal active inference agent (Cerebrum) for a household robot performing navigation, fetching, and sorting, optimizing for ultra-low computational operations and high activation sparsity.

Working directory: /Users/nazmi/cerebrum
Integrity mode: development

## Requirements

### R1. Household Chores Simulation Environment
Implement a simulated household environment in `benchmarks/tasks/household.py`. The environment must consist of a multi-room house layout (graph or grid-world) containing dynamic target objects (e.g. cup, book, trash) and target drop-off zones (e.g. table, shelf, bin). The task must support a sequence of:
1. Room Navigation: Mapping the house layout.
2. Object Identification: Locating target objects.
3. Object Fetch/Manipulate: Moving the agent to target objects, picking them up, and navigating to the target drop-off zone.
4. Sorting/Cleaning: Depositing objects in their correct slots.

### R2. Multimodal Active Inference Agent (Cerebrum)
Implement a closed-loop controller in the `CerebrumNet` framework where:
1. Sensory inputs are modeled as low-dimensional feature vectors representing pre-processed object and room identifiers.
2. Motor actions are generated internally using Active Inference (stochastic minimization of predictive error and free energy $F$).
3. The grid prior path-integration is driven by the agent's internally generated motor actions (efference copy).

### R3. Ultra-Low Computational Energy/Ops
Enforce and measure the following neuromorphic efficiency constraints:
1. **Activation Sparsity**: Average activation sparsity ($\rho$) in PC areas must be >= 80% during task execution.
2. **Operations Bound**: Synaptic operations per decision must be measured and logged.
3. **Communication**: Learn-time global communication must remain $O(1)$ scalar-only (neuromodulator $M$).

## Acceptance Criteria

### Verification
- [ ] Pytest suite passes fully (including all new household and active inference tests).
- [ ] Running `python3 benchmarks/run_household.py` verifies the active inference agent successfully completes chores (navigation, fetch, sort) with a task success rate >= 80% over 5 seeds.
- [ ] Average activation sparsity of PC areas is verified to be >= 80% across the benchmark runs.
- [ ] Energy logs verify that learn-time global communication remains $O(1)$ scalar-only.


