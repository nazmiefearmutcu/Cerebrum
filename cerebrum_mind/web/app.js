// Cerebrum-Mind OS Dashboard Logic

const API_BASE = ""; // Same origin

// App State
let activeRobot = null;
let robotsList = [];
let isTraining = false;
let isTaskRunning = false;

// Polling interval IDs
let trainingInterval = null;
let taskInterval = null;

// Chart history data
const maxChartPoints = 40;
let errorHistory = [];
let energyHistory = [];

// DOM Elements
const activeName = document.getElementById("active-name");
const activeClass = document.getElementById("active-class");
const activeDof = document.getElementById("active-dof");
const activeDimensions = document.getElementById("active-dimensions");
const activeSensors = document.getElementById("active-sensors");
const statusPulse = document.getElementById("status-pulse");
const statusText = document.getElementById("status-text");
const robotListContainer = document.getElementById("robot-list-container");

const valPcError = document.getElementById("val-pc-error");
const valFreeEnergy = document.getElementById("val-free-energy");
const valSynapseLock = document.getElementById("val-synapse-lock");
const valKpAlignment = document.getElementById("val-kp-alignment");

const activeKinErr = document.getElementById("active-kin-err");
const activeImpact = document.getElementById("active-impact");
const activeSlosh = document.getElementById("active-slosh");
const activeSlip = document.getElementById("active-slip");

const jointTelemetryContainer = document.getElementById("joint-telemetry-container");
const macrosContainer = document.getElementById("macros-container");
const adviceContainer = document.getElementById("advice-container");
const consoleOutput = document.getElementById("console-output");

const trainStartBtn = document.getElementById("train-start-btn");
const trainStopBtn = document.getElementById("train-stop-btn");
const sliderEta = document.getElementById("slider-eta");
const sliderTemp = document.getElementById("slider-temp");
const valEta = document.getElementById("val-eta");
const valTemp = document.getElementById("val-temp");

// Creator Modal Elements
const creatorDialog = document.getElementById("creator-dialog");
const openCreatorBtn = document.getElementById("open-creator-btn");
const closeCreatorX = document.getElementById("close-modal-x");
const closeCreatorBtn = document.getElementById("close-modal-btn");
const submitRobotBtn = document.getElementById("submit-robot-btn");

// Initialize Dashboard
window.addEventListener("DOMContentLoaded", () => {
  initApp();
  setupEventListeners();
  // Draw initial empty charts
  clearCharts();
});

// =========================================================================
// API Communications
// =========================================================================
async function initApp() {
  try {
    // 1. Fetch Robots list
    const response = await fetch(`${API_BASE}/api/robots`);
    const data = await response.json();
    robotsList = data.list;
    const activeNameStr = data.active;
    
    // Render Catalog
    renderCatalog();
    
    // Load Active Robot
    const robot = robotsList.find(r => r.name === activeNameStr);
    if (robot) {
      selectRobotProfile(robot);
    }
    
    // Load AI advice
    fetchAdvice();
    
    // Sync System status (in case server was restarted mid-run)
    const statusResp = await fetch(`${API_BASE}/api/status`);
    const statusData = await statusResp.json();
    if (statusData.training_active) {
      setTrainingUIState(true);
      startTrainingPolling();
    } else if (statusData.task_active) {
      setTaskUIState(true, "");
      startTaskPolling();
    }
    
  } catch (error) {
    writeLog(`[ERROR] Failed to connect to Cerebrum-Mind OS kernel: ${error.message}`, "error-line");
  }
}

function renderCatalog() {
  robotListContainer.innerHTML = "";
  robotsList.forEach(robot => {
    const item = document.createElement("div");
    item.className = `robot-item ${activeRobot && activeRobot.name === robot.name ? 'active' : ''}`;
    item.innerHTML = `
      <div class="robot-item-name">${robot.name}</div>
      <div class="robot-item-meta">
        <span>${robot.class}</span>
        <span>${robot.dof} DoF</span>
      </div>
    `;
    item.addEventListener("click", () => handleSelectRobot(robot.name));
    robotListContainer.appendChild(item);
  });
}

async function handleSelectRobot(name) {
  if (isTraining || isTaskRunning) {
    alert("Cannot switch robot profile while training or executing a task macro.");
    return;
  }
  
  try {
    const response = await fetch(`${API_BASE}/api/robots/select`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name })
    });
    const data = await response.json();
    if (data.success) {
      const robot = robotsList.find(r => r.name === name);
      selectRobotProfile(robot);
      writeLog(`[SYSTEM] Initialized ${name} profile control weights.`, "success-line");
      // Highlight in catalog
      document.querySelectorAll(".robot-item").forEach(item => {
        item.classList.remove("active");
        if (item.querySelector(".robot-item-name").innerText === name) {
          item.classList.add("active");
        }
      });
      fetchAdvice();
    } else {
      writeLog(`[ERROR] Selector failed: ${data.error}`, "error-line");
    }
  } catch (error) {
    writeLog(`[ERROR] Selector communication failure: ${error.message}`, "error-line");
  }
}

function selectRobotProfile(robot) {
  activeRobot = robot;
  activeName.innerText = robot.name;
  activeClass.innerText = robot.class;
  activeDof.innerText = `${robot.dof} DoF`;
  activeDimensions.innerText = `${robot.height} / ${robot.weight}`;
  
  // Sensors
  activeSensors.innerHTML = "";
  robot.sensors.forEach(s => {
    const chip = document.createElement("span");
    chip.className = "sensor-chip";
    chip.innerText = s;
    activeSensors.appendChild(chip);
  });
  
  // Render Macros
  renderMacros(robot.macros);
  
  // Initialize joint telemetry display
  initJointTelemetry(robot.joints);
  
  // Reset SVGs
  animateSkeleton({});
  
  // Clear charts
  clearCharts();
}

function renderMacros(macros) {
  macrosContainer.innerHTML = "";
  Object.keys(macros).forEach(key => {
    const m = macros[key];
    const btn = document.createElement("button");
    btn.className = "macro-btn";
    btn.id = `macro-btn-${key}`;
    btn.innerHTML = `
      <span class="macro-icon">${m.icon}</span>
      <div class="macro-info">
        <span class="macro-name">${m.name}</span>
        <span class="macro-desc">${m.description}</span>
      </div>
    `;
    btn.addEventListener("click", () => triggerMacro(key));
    macrosContainer.appendChild(btn);
  });
}

function initJointTelemetry(joints) {
  jointTelemetryContainer.innerHTML = "";
  joints.forEach(j => {
    const row = document.createElement("div");
    row.className = "joint-telemetry-row";
    row.id = `joint-row-${j}`;
    row.innerHTML = `
      <span class="joint-name">${j}</span>
      <span class="joint-angle" id="angle-${j}">0.00°</span>
      <span class="joint-error clean" id="error-${j}">0.000</span>
    `;
    jointTelemetryContainer.appendChild(row);
  });
}

// =========================================================================
// Task Exec / Macros
// =========================================================================
async function triggerMacro(key) {
  if (isTraining) {
    alert("Halt core training before executing operational task macros.");
    return;
  }
  
  if (isTaskRunning) {
    // Stop running task
    try {
      const response = await fetch(`${API_BASE}/api/task/stop`, { method: "POST" });
      const data = await response.json();
      if (data.success) {
        setTaskUIState(false);
        stopTaskPolling();
        writeLog("[SYSTEM] Task execution manually aborted.", "error-line");
      }
    } catch (e) {
      writeLog(`[ERROR] Failed to stop macro: ${e.message}`, "error-line");
    }
  } else {
    // Start task
    try {
      const response = await fetch(`${API_BASE}/api/task/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ task: key })
      });
      const data = await response.json();
      if (data.success) {
        setTaskUIState(true, key);
        startTaskPolling();
        writeLog(`[SYSTEM] Starting execution of macro: ${activeRobot.macros[key].name}`, "success-line");
      } else {
        writeLog(`[ERROR] Macro launch failed: ${data.error}`, "error-line");
      }
    } catch (e) {
      writeLog(`[ERROR] Macro launch communication failure: ${e.message}`, "error-line");
    }
  }
}

function setTaskUIState(running, key = "") {
  isTaskRunning = running;
  
  // Disable training buttons
  trainStartBtn.disabled = running;
  
  // Disable selection
  document.querySelectorAll(".robot-item").forEach(el => {
    if (running) el.style.pointerEvents = "none";
    else el.style.pointerEvents = "auto";
  });
  
  if (running) {
    statusPulse.className = "pulse-indicator active";
    statusText.innerText = "Task Executing";
    
    // Alter active button UI
    document.querySelectorAll(".macro-btn").forEach(btn => {
      btn.disabled = true;
      if (btn.id === `macro-btn-${key}`) {
        btn.disabled = false;
        btn.classList.add("running");
        btn.querySelector(".macro-desc").innerText = "Executing Macro... Click to abort.";
      }
    });
  } else {
    statusPulse.className = "pulse-indicator online";
    statusText.innerText = "Core Ready";
    
    // Restore buttons
    document.querySelectorAll(".macro-btn").forEach(btn => {
      btn.disabled = false;
      btn.classList.remove("running");
      const btnKey = btn.id.replace("macro-btn-", "");
      if (activeRobot && activeRobot.macros[btnKey]) {
        btn.querySelector(".macro-desc").innerText = activeRobot.macros[btnKey].description;
      }
    });
  }
}

function startTaskPolling() {
  if (taskInterval) clearInterval(taskInterval);
  
  // Clear charts
  errorHistory = [];
  energyHistory = [];
  
  taskInterval = setInterval(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/task/status`);
      const data = await response.json();
      
      // Update Vitals
      valPcError.innerText = data.pc_error.toFixed(4);
      valFreeEnergy.innerText = data.free_energy.toFixed(4);
      
      // Update Physics telemetry
      if (activeKinErr) activeKinErr.innerText = `${data.kinematics_error.toFixed(3)}m`;
      if (activeImpact) activeImpact.innerText = `${data.impact_g_force.toFixed(2)} G`;
      if (activeSlosh) activeSlosh.innerText = data.fluid_slosh_index.toFixed(3);
      if (activeSlip) activeSlip.innerText = `${data.wheel_slip_drift.toFixed(2)} m/s`;
      
      // Update charts
      pushHistory(data.pc_error, data.free_energy);
      
      // Update Active macro progress
      const activeBtn = document.querySelector(".macro-btn.running");
      if (activeBtn) {
        activeBtn.querySelector(".macro-desc").innerText = `Phase: ${data.phase} (${data.completion_pct}%) - Click to abort.`;
      }
      
      // Print logs
      if (data.logs && data.logs.length > 0) {
        // Find new lines
        const lastLine = data.logs[data.logs.length - 1];
        writeLog(lastLine);
      }
      
      // Update joint lists and SVG
      updateJointsUI(data.joint_angles, data.joint_errors);
      animateSkeleton(data.joint_angles);
      
      // Check termination
      if (!isTaskRunning || data.completion_pct >= 100) {
        clearInterval(taskInterval);
        setTaskUIState(false);
        fetchAdvice();
      }
    } catch (e) {
      writeLog(`[ERROR] Task polling error: ${e.message}`, "error-line");
      clearInterval(taskInterval);
      setTaskUIState(false);
    }
  }, 400);
}

function stopTaskPolling() {
  if (taskInterval) {
    clearInterval(taskInterval);
    taskInterval = null;
  }
}

function updateJointsUI(angles, errors) {
  if (!angles || !errors) return;
  Object.keys(angles).forEach(j => {
    const angleEl = document.getElementById(`angle-${j}`);
    const errEl = document.getElementById(`error-${j}`);
    const rowEl = document.getElementById(`joint-row-${j}`);
    
    if (angleEl && errEl) {
      // Rad to degrees
      const deg = angles[j] * (180 / Math.PI);
      angleEl.innerText = `${deg.toFixed(1)}°`;
      
      const err = errors[j];
      errEl.innerText = err.toFixed(3);
      
      // Alert categories for error levels
      errEl.className = "joint-error";
      if (err > 0.04) {
        errEl.classList.add("severe");
        if (rowEl) rowEl.style.borderColor = "var(--color-error)";
      } else if (err > 0.02) {
        errEl.classList.add("warn");
        if (rowEl) rowEl.style.borderColor = "var(--color-warning)";
      } else {
        errEl.classList.add("clean");
        if (rowEl) rowEl.style.borderColor = "rgba(255,255,255,0.05)";
      }
    }
  });
}

// =========================================================================
// Training Controls
// =========================================================================
setupEventListeners = () => {
  // Sliders
  sliderEta.addEventListener("input", (e) => {
    valEta.innerText = parseFloat(e.target.value).toFixed(3);
  });
  sliderTemp.addEventListener("input", (e) => {
    valTemp.innerText = parseFloat(e.target.value).toFixed(2);
  });
  
  trainStartBtn.addEventListener("click", startTraining);
  trainStopBtn.addEventListener("click", stopTraining);
  
  // Custom creator modal toggles
  openCreatorBtn.addEventListener("click", () => {
    creatorDialog.showModal();
  });
  
  const closeModal = () => {
    creatorDialog.close();
  };
  
  closeCreatorX.addEventListener("click", closeModal);
  closeCreatorBtn.addEventListener("click", closeModal);
  
  // Register form submit
  creatorDialog.querySelector("form").addEventListener("submit", handleRegisterCustomRobot);
};

async function startTraining() {
  try {
    const response = await fetch(`${API_BASE}/api/train/start`, { method: "POST" });
    const data = await response.json();
    if (data.success) {
      setTrainingUIState(true);
      startTrainingPolling();
      writeLog("[SYSTEM] Initiating training loop of Cerebrum PC Modules.", "success-line");
    } else {
      writeLog(`[ERROR] Start training failed: ${data.error}`, "error-line");
    }
  } catch (e) {
    writeLog(`[ERROR] Failed to contact training kernel: ${e.message}`, "error-line");
  }
}

async function stopTraining() {
  try {
    const response = await fetch(`${API_BASE}/api/train/stop`, { method: "POST" });
    const data = await response.json();
    if (data.success) {
      setTrainingUIState(false);
      stopTrainingPolling();
      writeLog("[SYSTEM] Training loop terminated. Motor synapse values consolidated.", "success-line");
      fetchAdvice();
    }
  } catch (e) {
    writeLog(`[ERROR] Failed to stop training: ${e.message}`, "error-line");
  }
}

function setTrainingUIState(running) {
  isTraining = running;
  trainStartBtn.disabled = running;
  trainStopBtn.disabled = !running;
  
  // Disable macro buttons
  document.querySelectorAll(".macro-btn").forEach(btn => {
    btn.disabled = running;
  });
  
  // Disable robot catalog
  document.querySelectorAll(".robot-item").forEach(el => {
    if (running) el.style.pointerEvents = "none";
    else el.style.pointerEvents = "auto";
  });
  
  if (running) {
    statusPulse.className = "pulse-indicator active";
    statusText.innerText = "Training Core";
  } else {
    statusPulse.className = "pulse-indicator online";
    statusText.innerText = "Core Ready";
  }
}

function startTrainingPolling() {
  if (trainingInterval) clearInterval(trainingInterval);
  
  // Reset charts
  errorHistory = [];
  energyHistory = [];
  let stepsCounter = 0;
  
  trainingInterval = setInterval(async () => {
    try {
      const response = await fetch(`${API_BASE}/api/train/status`);
      const data = await response.json();
      
      // Update UI vitals
      valPcError.innerText = data.pc_error.toFixed(4);
      valFreeEnergy.innerText = data.free_energy.toFixed(4);
      valSynapseLock.innerText = `${data.synapses_locked_pct.toFixed(1)}%`;
      valKpAlignment.innerText = data.kp_alignment.toFixed(2);
      
      // Update training wheel slip
      if (activeSlip) activeSlip.innerText = `${data.wheel_slip_drift.toFixed(2)} m/s`;
      
      // Push history
      pushHistory(data.pc_error, data.free_energy);
      
      // Print new logs
      if (data.log && data.log.length > 0 && stepsCounter % 5 === 0) {
        const lastLine = data.log[data.log.length - 1];
        writeLog(lastLine);
      }
      
      // Periodically get AI advice
      if (stepsCounter % 15 === 0) {
        fetchAdvice();
      }
      
      stepsCounter++;
    } catch (e) {
      writeLog(`[ERROR] Training polling failed: ${e.message}`, "error-line");
      clearInterval(trainingInterval);
      setTrainingUIState(false);
    }
  }, 300);
}

function stopTrainingPolling() {
  if (trainingInterval) {
    clearInterval(trainingInterval);
    trainingInterval = null;
  }
}

// =========================================================================
// AI Diagnostics / Advice
// =========================================================================
async function fetchAdvice() {
  try {
    const response = await fetch(`${API_BASE}/api/advice`);
    const list = await response.json();
    
    adviceContainer.innerHTML = "";
    list.forEach(a => {
      const card = document.createElement("div");
      card.className = `advice-card ${a.severity}`;
      card.innerHTML = `
        <div class="advice-cat">${a.category} (${a.severity})</div>
        <div class="advice-msg">${a.message}</div>
        <div class="advice-rec">💡 <strong>Recommendation:</strong> ${a.recommendation}</div>
      `;
      adviceContainer.appendChild(card);
    });
  } catch (e) {
    console.error("Failed to load advice: ", e);
  }
}

// =========================================================================
// Custom Robot Registration
// =========================================================================
async function handleRegisterCustomRobot(e) {
  e.preventDefault();
  
  const name = document.getElementById("robot-name-input").value.trim();
  const robotClass = document.getElementById("robot-class-select").value;
  const height = document.getElementById("robot-height-input").value.trim() || "N/A";
  const weight = document.getElementById("robot-weight-input").value.trim() || "N/A";
  
  // Read sensors checkboxes
  const sensorCheckboxes = document.querySelectorAll("input[name='sensors']:checked");
  const sensors = Array.from(sensorCheckboxes).map(cb => cb.value);
  
  // Parse joints
  const jointsText = document.getElementById("robot-joints-input").value;
  const joints = jointsText.split(/[,\s]+/).map(j => j.trim()).filter(j => j.length > 0);
  
  if (joints.length === 0) {
    alert("Please define at least one actuator joint.");
    return;
  }
  
  const payload = {
    name,
    class: robotClass,
    height,
    weight,
    sensors,
    joints
  };
  
  try {
    const response = await fetch(`${API_BASE}/api/robots/custom`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    
    const data = await response.json();
    if (data.success) {
      writeLog(`[SYSTEM] Registered Custom Robot profile: ${name}`, "success-line");
      creatorDialog.close();
      
      // Clear inputs
      document.getElementById("robot-name-input").value = "";
      document.getElementById("robot-joints-input").value = "";
      
      // Reload catalog
      initApp();
    } else {
      alert(`Registration failed: ${data.error}`);
    }
  } catch (error) {
    alert(`Registration communication error: ${error.message}`);
  }
}

// =========================================================================
// Helpers (Logger & Charts Draw)
// =========================================================================
function writeLog(text, className = "system-line") {
  const line = document.createElement("span");
  line.className = className;
  line.innerText = text;
  consoleOutput.appendChild(line);
  consoleOutput.scrollTop = consoleOutput.scrollHeight;
}

function pushHistory(err, nrg) {
  errorHistory.push(err);
  energyHistory.push(nrg);
  
  if (errorHistory.length > maxChartPoints) errorHistory.shift();
  if (energyHistory.length > maxChartPoints) energyHistory.shift();
  
  renderCharts();
}

function clearCharts() {
  errorHistory = Array(maxChartPoints).fill(0.0);
  energyHistory = Array(maxChartPoints).fill(0.0);
  renderCharts();
}

function renderCharts() {
  drawChart("chart-error", errorHistory, "#00f2fe", "rgba(0,242,254,0.1)");
  drawChart("chart-energy", energyHistory, "#7f00ff", "rgba(127,0,255,0.1)");
}

function drawChart(canvasId, data, lineColor, fillColor) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  
  ctx.clearRect(0, 0, width, height);
  
  if (data.length === 0) return;
  
  // Find ranges
  let max = Math.max(...data, 0.1);
  let min = Math.min(...data, 0.0);
  let range = max - min;
  
  // Draw Grid Lines
  ctx.strokeStyle = "rgba(255,255,255,0.05)";
  ctx.lineWidth = 1;
  for (let i = 1; i < 4; i++) {
    const y = (height / 4) * i;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
  
  // Draw Graph Curve
  ctx.beginPath();
  const getX = (idx) => (width / (maxChartPoints - 1)) * idx;
  const getY = (val) => height - 10 - ((val - min) / range) * (height - 20);
  
  ctx.moveTo(getX(0), getY(data[0]));
  for (let i = 1; i < data.length; i++) {
    ctx.lineTo(getX(i), getY(data[i]));
  }
  
  // Save line path for drawing it
  ctx.strokeStyle = lineColor;
  ctx.lineWidth = 2.5;
  ctx.shadowColor = lineColor;
  ctx.shadowBlur = 8;
  ctx.stroke();
  
  // Reset shadow for fill
  ctx.shadowBlur = 0;
  
  // Fill under curve
  ctx.lineTo(getX(data.length - 1), height);
  ctx.lineTo(getX(0), height);
  ctx.closePath();
  ctx.fillStyle = fillColor;
  ctx.fill();
  
  // Print Text Labels (min/max)
  ctx.fillStyle = "rgba(255,255,255,0.4)";
  ctx.font = "9px Space Mono";
  ctx.fillText(max.toFixed(3), 5, 12);
  ctx.fillText(min.toFixed(3), 5, height - 4);
}

// =========================================================================
// SVG Robot skeleton animation (Forward Kinematics simulator)
// =========================================================================
function animateSkeleton(angles) {
  const getAngle = (j, defVal = 0.0) => {
    return (angles && angles[j] !== undefined) ? angles[j] : defVal;
  };
  
  // 1. Head Rotation (neck_yaw / neck_pitch)
  const neckYaw = getAngle("neck_yaw");
  const neckPitch = getAngle("neck_pitch");
  const headCircle = document.getElementById("joint-head");
  if (headCircle) {
    headCircle.setAttribute("cx", 100 + Math.sin(neckYaw) * 12);
    headCircle.setAttribute("cy", 70 + Math.cos(neckPitch) * 8 - 8);
  }
  
  // Kinematics for arms
  // Left arm starting joint (Shoulder) (60, 120)
  const lShoulderPitch = getAngle("left_shoulder_pitch", 0.3) + getAngle("l_shoulder_pitch", 0.0);
  const lShoulderRoll = getAngle("left_shoulder_roll", 0.0) + getAngle("l_shoulder_roll", 0.0);
  const lElbow = getAngle("left_elbow", 0.4) + getAngle("l_elbow", 0.0);
  
  const lx2 = 60 - Math.sin(lShoulderPitch) * 45;
  const ly2 = 120 + Math.cos(lShoulderPitch) * 45;
  
  const lx3 = lx2 - Math.sin(lShoulderPitch + lElbow) * 35;
  const ly3 = ly2 + Math.cos(lShoulderPitch + lElbow) * 35;
  
  // Update Left Arm lines & nodes
  setLineCoords("link-l-arm-upper", 60, 120, lx2, ly2);
  setLineCoords("link-l-arm-lower", lx2, ly2, lx3, ly3);
  setCircleCoords("joint-l-shoulder", 60, 120);
  setCircleCoords("joint-l-elbow", lx2, ly2);
  setCircleCoords("joint-l-wrist", lx3, ly3);

  // Right arm starting joint (Shoulder) (140, 120)
  const rShoulderPitch = getAngle("right_shoulder_pitch", 0.3) + getAngle("r_shoulder_pitch", 0.0);
  const rElbow = getAngle("right_elbow", 0.4) + getAngle("r_elbow", 0.0);
  
  const rx2 = 140 + Math.sin(rShoulderPitch) * 45;
  const ry2 = 120 + Math.cos(rShoulderPitch) * 45;
  
  const rx3 = rx2 + Math.sin(rShoulderPitch + rElbow) * 35;
  const ry3 = ry2 + Math.cos(rShoulderPitch + rElbow) * 35;
  
  // Update Right Arm
  setLineCoords("link-r-arm-upper", 140, 120, rx2, ry2);
  setLineCoords("link-r-arm-lower", rx2, ry2, rx3, ry3);
  setCircleCoords("joint-r-shoulder", 140, 120);
  setCircleCoords("joint-r-elbow", rx2, ry2);
  setCircleCoords("joint-r-wrist", rx3, ry3);

  // Kinematics for legs
  // Left Leg Hip start (75, 220)
  const lHipPitch = getAngle("left_hip_pitch", 0.0) + getAngle("l_hip", 0.0);
  const lKnee = getAngle("left_knee", 0.0) + getAngle("l_knee", 0.0);
  const lAnkle = getAngle("left_ankle", 0.0) + getAngle("l_ankle", 0.0);
  
  const llx2 = 75 - Math.sin(lHipPitch) * 60;
  const lly2 = 220 + Math.cos(lHipPitch) * 60;
  
  const llx3 = llx2 - Math.sin(lHipPitch - lKnee) * 55;
  const lly3 = lly2 + Math.cos(lHipPitch - lKnee) * 55;
  
  // Update Left Leg
  setLineCoords("link-l-leg-upper", 75, 220, llx2, lly2);
  setLineCoords("link-l-leg-lower", llx2, lly2, llx3, lly3);
  setCircleCoords("joint-l-hip", 75, 220);
  setCircleCoords("joint-l-knee", llx2, lly2);
  setCircleCoords("joint-l-ankle", llx3, lly3);

  // Right Leg Hip start (125, 220)
  const rHipPitch = getAngle("right_hip_pitch", 0.0) + getAngle("r_hip", 0.0);
  const rKnee = getAngle("right_knee", 0.0) + getAngle("r_knee", 0.0);
  
  const rlx2 = 125 + Math.sin(rHipPitch) * 60;
  const rly2 = 220 + Math.cos(rHipPitch) * 60;
  
  const rlx3 = rlx2 + Math.sin(rHipPitch - rKnee) * 55;
  const rly3 = rly2 + Math.cos(rHipPitch - rKnee) * 55;
  
  // Update Right Leg
  setLineCoords("link-r-leg-upper", 125, 220, rlx2, rly2);
  setLineCoords("link-r-leg-lower", rlx2, rly2, rlx3, rly3);
  setCircleCoords("joint-r-hip", 125, 220);
  setCircleCoords("joint-r-knee", rlx2, rly2);
  setCircleCoords("joint-r-ankle", rlx3, rly3);
}

function setLineCoords(id, x1, y1, x2, y2) {
  const line = document.getElementById(id);
  if (line) {
    line.setAttribute("x1", x1);
    line.setAttribute("y1", y1);
    line.setAttribute("x2", x2);
    line.setAttribute("y2", y2);
  }
}

function setCircleCoords(id, cx, cy) {
  const circle = document.getElementById(id);
  if (circle) {
    circle.setAttribute("cx", cx);
    circle.setAttribute("cy", cy);
  }
}
