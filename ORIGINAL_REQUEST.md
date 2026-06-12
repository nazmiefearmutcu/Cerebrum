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


## Follow-up — 2026-06-11T18:22:57Z

Addressing the three core deficiencies in the Cerebrum project: PyTorch GPU acceleration backend, Sensory-Motor Grounding with PyBullet/ROS 2 wrapper, and System 1 (Cerebellum) reflex bypass.

Working directory: /Users/nazmi/Cerebrum
Integrity mode: development

## Requirements

### R1. PyTorch Acceleration Backend
- Port the core CerebrumNet architecture (and modules `PCAreas`, `GridHead`, `BasalGangliaGate`, `Workspace`, `Neuromodulator`, `MetaplasticFuse`) from NumPy to PyTorch.
- Preserve all mathematical invariants: backprop-free updates, local weight updates, weight transport bans, and scalar-M modulation.
- Support device-agnostic execution (`cpu`, `cuda`, `mps`).

### R2. Sensory-Motor Grounding & PyBullet/ROS 2 Integration
- Implement sensory processors (e.g., converting continuous camera/depth/lidar readings to vector states) and motor processors (e.g., converting workspace vectors to control targets).
- Create a simulator interface using **PyBullet** to model a physical robot (e.g., a simple cartpole, mobile robot, or gripper) interacting in a continuous 3D physical environment.
- Create a ROS 2 node wrapper (using `rclpy` or a robust mock interface if ROS 2 is not installed locally) exposing topics for sensory input and motor command output.

### R3. System 1 (Cerebellum) Reflex Bypass
- Implement a fast-path reflex module (System 1) that bypasses the slow multi-step workspace settling of CerebrumNet (System 2) when high-urgency states (e.g., imminent collision, loss of balance, or high-energy prediction errors) are detected.
- System 1 should directly map critical sensory states to immediate motor commands to stabilize the robot or avoid obstacles.

## Acceptance Criteria

### Unit and Integration Tests
- [ ] PyTorch port achieves 100% equivalence in learning and inference steps compared to the original NumPy version (under matching seeds, within float tolerance).
- [ ] All existing test suites pass under the PyTorch backend, and new tests verify device-agnostic execution on `cpu` or `cuda`/`mps` when available.

### Robot Simulation & Grounding
- [ ] PyBullet simulation runs continuously and connects to CerebrumNet's sensory-motor pipeline.
- [ ] The ROS 2/mock-ROS 2 node successfully publishes/subscribes to `/sensory_input` and `/motor_commands`.
- [ ] The grounded CerebrumNet agent successfully accomplishes a basic physical navigation/control task (e.g., reaching a target position) in the PyBullet environment.

### System 1 Reflex Performance
- [ ] Verification tests show that System 1 latency is at least 5x lower (fewer settling iterations/ops) than System 2 workspace routing when a hazard is detected.
- [ ] In obstacle-avoidance simulations, the robot equipped with System 1 reflexes successfully avoids collisions that a System 2-only model fails to avoid in time due to settling delay.

## Follow-up — 2026-06-12T19:36:16Z

Cerebrum aktif çıkarım mimarisinin rapor edilen tüm mimari eksikliklerini (C3 kaldıraç rekabeti, metrik olmayan graflar, metaplastik sigorta kararsızlığı, yapay enerji sapması vb.) gidermek ve önerilen çözümleri (altuzay ayrıştırması, topolojiye duyarlı metrik olmayan önsel dönüşümleri, dinamik homeostatik durulma ve ROS node senkronizasyonu) implemente ederek sistemi kararlı hale getirmek.

Working directory: /Users/nazmi/Cerebrum
Integrity mode: development

## Requirements

### R1. Grid Önseli İçin Altuzay Ayrıştırması (Subspace Segregation)
Kortikal modüllerin (`PCAreas`) üst katmanındaki top-down grid tahmini ile duyusal faktör kodlarının birbiriyle yarışmasını önlemek amacıyla boyutsal altuzay ayrıştırması (subspace segregation) uygulayın. Grid önsel tahminleri ve duyusal kodlar latent vektör içinde çakışmayacak şekilde izole edilmelidir.

### R2. Komütatif Olmayan ve Asimetrik Yapısal Önsel (Non-Commutative/Asymmetric Prior)
Grid hücresi modelini (`GridHead`) ya da alternatif bir önsel modülü, değişmeli (commutative) olmayan Lie-grubu dönüşümlerini (örn. $SO(3)$ veya Heisenberg grupları) destekleyecek şekilde genişletin. Bu sayede yönlü ağaçlar (directed trees), hiyerarşiler ve asimetrik ilişkiler içeren graflar (TreeRelationalGraph) üzerinde başarılı path-integration yapılabilmelidir (FM7 çözümü).

### R3. Metaplastik Homeostaz ve Kararlılık (Metaplastic Homeostasis)
Metaplastic sigorta (`MetaplasticFuse`) için konsolidasyon rezervini ($c$) uzun eğitim süreçlerinde ($\ge 200$ adım) kararlı kılan ve Langevin gürültüsünün sürpriz olarak algılanıp rezervi eritmesini önleyen dinamik bir homeostaz veya sıcaklık adaptasyonu mekanizması entegre edin (FM4 çözümü).

### R4. ROS Node Senkronizasyonu ve Sistem 1-2 Geçiş Yumuşatması
System 1 (Refleks) tetiklendiğinde System 2 (Workspace durulması) arasındaki motor komutu geçişlerini yumuşatın (sarsıntıyı engellemek için). Ayrıca, `ros_node.py` içindeki asenkron motor yazma ve hız okuma işlemlerindeki thread çekişmelerini (race hazards) tamamen engelleyin.

### R5. Kolen-Pollack Eşleşme ve Gumbel-Max Kararlılığı
Ağırlık güncelleme dinamiğinde ileri ağırlıklar $W$ ile geri besleme ağırlıkları $B$ arasındaki Kolen-Pollack (KP) güncelleme uyumsuzluklarını giderin. Modüllerin gating yarışında ("dead experts" sorunu) homeostaz parametrelerinin kararlılığını artırın.

## Acceptance Criteria

### Doğruluk ve Entegrasyon
- [ ] Bütün pytest test paketi (yeni eklenecek testlerle birlikte) %100 yeşil (başarılı) olmalıdır.
- [ ] Precision-balancing / subspace özelliği etkinleştirilmiş `full` koşulu altındaki faktörize kod deşifre doğruluğu (trained decode accuracy) >= 0.85 olmalıdır (mevcut çöküş değeri olan ~0.28'den yukarı taşınmalıdır).
- [ ] Metrik olmayan asimetrik graflardaki (TreeRelationalGraph) few-shot tahmin başarısı, flat-prior baz hattına göre belirgin şekilde yüksek çıkmalıdır.
- [ ] Task A görevinin unutulma oranı, Task C'nin öğrenilmesinden sonra (200+ eğitim adımı geçildiğinde dahi) 0.15'in altında tutulmalıdır.
- [ ] ROS node telemetrisi sıfır race condition ve System 1 tetiklenmelerinde yumuşak hız komut geçişlerini doğrulamalıdır.
