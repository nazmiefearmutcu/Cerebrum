from dataclasses import dataclass, field

@dataclass(frozen=True)
class CerebrumConfig:
    # PC hierarchy
    dims: tuple = (16, 12, 8)        # area sizes, dims[0] = observation/lowest area
    # timescales (smaller = faster)
    tau_x: float = 1.0               # activity settling
    tau_e: float = 5.0               # eligibility trace
    tau_w: float = 200.0             # weight plasticity (slow)
    tau_pi: float = 100.0            # precision learning
    tau_r: float = 50.0              # reward baseline EMA
    tau_b: float = 200.0             # feedback weight learning
    # settling
    dt: float = 0.1
    n_settle: int = 40
    T_floor: float = 0.02            # Pillar 4: noise floor > 0 forbids MAP collapse
    T0: float = 0.2                  # initial annealing temperature
    tau_anneal: float = 15.0
    # costs / rates
    gamma: float = 0.01              # activity L1 sparsity (R(x))
    eta_w: float = 0.02              # weight learning rate scale
    eta_b: float = 0.01              # feedback weight learning rate
    lam_b: float = 1e-3              # feedback weight decay
    # --- Kolen-Pollack feedback alignment (OPT-IN; default OFF = behavior unchanged) ---
    align_feedback: bool = False     # if True, B is driven toward W.T by a matched LOCAL rule
                                     # (KP): B and W receive the SAME M-gated pre*post product
                                     # (transposed) with a MATCHED decay, so (W - B.T) -> 0
                                     # WITHOUT ever reading/copying W.T (no weight transport).
    lam_kp: float = 1e-2             # matched symmetric weight decay applied to BOTH W and B
                                     # ONLY when align_feedback=True (the KP coupling term).
    # --- top-down precision balancing (OPT-IN; default OFF = behavior unchanged) ---
    balance_grid_precision: bool = False
                                     # if True, the EXTERNAL top-down prediction at the TOP area
                                     # (the grid HEAD's structural prediction in CerebrumNet/CerebrumCore)
                                     # is gain-normalized to the bottom-up activity scale of that
                                     # area BEFORE it enters the top-area error. In predictive coding
                                     # the relative pull of a top-down prediction is set by PRECISION;
                                     # the never-decayed Hebbian grid content store makes ||top_pred||
                                     # ~500x the small obs-driven latent (|x|~0.1), so its unit-precision
                                     # pull CRUSHES the obs factor code (the latent tracks grid phase,
                                     # not obs factors). This LOCAL, per-area diagonal gain rescales
                                     # the prediction so the grid top-down and the bottom-up
                                     # reconstruction signal are weighted COMPARABLY. NO global
                                     # objective, no weight transport — a pure prediction-gain op.
    grid_precision_ref: float = 1.0  # target ratio ||scaled top_pred|| / ||bottom-up activity||;
                                     # 1.0 = match the bottom-up scale exactly (only used when
                                     # balance_grid_precision=True).
    Pi0: float = 1.0                 # precision prior
    sigma0: float = 1.0              # precision floor variance
    kappa_pi: float = 1.0            # precision learning gain
    # grid HEAD
    grid_n_modules: int = 6
    grid_lambda0: float = 4.0        # base spatial period
    grid_ratio: float = 1.42         # geometric module scaling
    grid_eta_bind: float = 1.0       # content-store binding rate
    # gate / workspace (Stage 2)
    lam_g: float = 0.0        # gate Go/NoGo weight decay toward init (0 = off; >0 prevents spurious
                              # preference drift when there is no stable per-module target to learn)
    gate_temp: float = 0.0    # fixed gate selection temperature (0 = unset -> use neuromodulator 1/M);
                              # a low value lets the informative scalar bid dominate (still stochastic)
    # metaplasticity (Stage 3)
    tau_S: float = 20.0       # surprise-baseline EMA timescale
    tau_c: float = 300.0      # consolidation-reserve timescale (slow)
    alpha_c: float = 1.0      # low-surprise consolidation gain (builds c)
    beta_c: float = 1.5       # high-surprise erosion gain (frees c)
    c_max: float = 1.0        # max consolidation reserve
    g_theta: float = 4.0      # plasticity-permission sigmoid sharpness
    # misc
    pc_sparsity_threshold: float = 0.0
    seed: int = 0
