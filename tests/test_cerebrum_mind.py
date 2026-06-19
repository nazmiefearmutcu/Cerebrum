import os
import json
import pytest
from cerebrum_mind.robot_configs import get_all_robots, save_custom_robot, load_custom_robots
from cerebrum_mind.core import CerebrumMindOS

def test_robot_configs_loading():
    """Verify standard robot profiles load correctly."""
    robots = get_all_robots()
    assert "Unitree G1" in robots
    assert "Optimus Gen 2" in robots
    assert "BD Atlas" in robots
    assert "Figure 01" in robots
    
    g1 = robots["Unitree G1"]
    assert g1["class"] == "Humanoid"
    assert g1["dof"] == 23
    assert "3D LiDAR" in g1["sensors"]
    assert "clean_house" in g1["macros"]

def test_custom_robot_save_and_load(tmp_path):
    """Verify custom robot profile creation and JSON serialization."""
    # Override CONFIG_FILE path for testing
    import cerebrum_mind.robot_configs as rc
    original_file = rc.CUSTOM_CONFIG_FILE
    
    test_file = os.path.join(tmp_path, "test_custom_robots.json")
    rc.CUSTOM_CONFIG_FILE = test_file
    
    custom_config = {
        "name": "Test Bot 9000",
        "class": "Quadruped",
        "height": "0.6 m",
        "weight": "12 kg",
        "joints": ["front_left_hip", "front_left_knee", "rear_right_hip", "rear_right_knee"],
        "sensors": ["IMU", "Camera"]
    }
    
    try:
        # Save custom robot
        success = save_custom_robot(custom_config)
        assert success is True
        
        # Load and verify
        all_bots = get_all_robots()
        assert "Test Bot 9000" in all_bots
        
        bot = all_bots["Test Bot 9000"]
        assert bot["class"] == "Quadruped"
        assert bot["dof"] == 4
        assert "Camera" in bot["sensors"]
        assert "custom_task" in bot["macros"]
    finally:
        # Restore configuration paths
        rc.CUSTOM_CONFIG_FILE = original_file

def test_mind_os_initialization():
    """Verify CerebrumMindOS core initializes with correct defaults."""
    os_instance = CerebrumMindOS()
    status = os_instance.get_status()
    
    assert status["active_robot"] == "Unitree G1"
    assert status["active_robot_class"] == "Humanoid"
    assert status["dof"] == 23
    assert "3D LiDAR" in status["sensors"]
    assert "clean_house" in status["macros"]
    assert status["training_active"] is False
    assert status["task_active"] is False

def test_mind_os_switch_profile():
    """Verify we can hot-swap robot profiles in the OS manager."""
    os_instance = CerebrumMindOS()
    
    # Switch to Optimus
    success, msg = os_instance.set_active_robot("Optimus Gen 2")
    assert success is True
    
    status = os_instance.get_status()
    assert status["active_robot"] == "Optimus Gen 2"
    assert status["dof"] == 39
    
    # Try invalid robot
    success, msg = os_instance.set_active_robot("Terminator T-800")
    assert success is False
    assert status["active_robot"] == "Optimus Gen 2"  # Unchanged

def test_ai_advisor_generation():
    """Verify that the AI Advisor generates meaningful tips."""
    os_instance = CerebrumMindOS()
    
    # Before training (idle state advice)
    advice = os_instance.get_ai_advice()
    assert len(advice) > 0
    categories = [a["category"] for a in advice]
    assert "Core OS Vitals" in categories
    
    # Mock elevated error to trigger diagnostics
    os_instance.training_metrics["step"] = 50
    os_instance.training_metrics["pc_error"] = 2.5
    
    advice = os_instance.get_ai_advice()
    categories = [a["category"] for a in advice]
    assert "Predictive Coding" in categories
    
    # Find warning warning advice
    warning_advice = [a for a in advice if a["category"] == "Predictive Coding"][0]
    assert warning_advice["severity"] == "Caution"
    assert "n_settle" in warning_advice["recommendation"]

def test_server_api_endpoints():
    """Verify the HTTP server starts, serves static files, and responds to API calls."""
    import socket
    import urllib.request
    from threading import Thread
    from socketserver import ThreadingTCPServer
    from cerebrum_mind.server import CerebrumMindHTTPHandler

    # Find a free port
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()

    # Start the server in a background thread
    server = ThreadingTCPServer(("127.0.0.1", port), CerebrumMindHTTPHandler)
    server_thread = Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        # 1. Test Static files serving
        url_static = f"http://127.0.0.1:{port}/"
        with urllib.request.urlopen(url_static) as response:
            assert response.status == 200
            content = response.read().decode("utf-8")
            assert "CEREBRUM-MIND" in content
            assert "app.js" in content
            
        # 2. Test GET robots API
        url_robots = f"http://127.0.0.1:{port}/api/robots"
        with urllib.request.urlopen(url_robots) as response:
            assert response.status == 200
            data = json.loads(response.read().decode("utf-8"))
            assert "active" in data
            assert data["active"] == "Unitree G1"
            assert len(data["list"]) >= 4

        # 3. Test GET advice API
        url_advice = f"http://127.0.0.1:{port}/api/advice"
        with urllib.request.urlopen(url_advice) as response:
            assert response.status == 200
            advice_data = json.loads(response.read().decode("utf-8"))
            assert len(advice_data) > 0
            
    finally:
        # Shut down server cleanly
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=1.0)

