import os
import sys
import json
import urllib.parse
from http.server import BaseHTTPRequestHandler
from socketserver import ThreadingTCPServer

# Ensure cerebrum modules can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cerebrum_mind.core import CerebrumMindOS
from cerebrum_mind.robot_configs import save_custom_robot, get_all_robots

# Initialize global OS manager
os_manager = CerebrumMindOS()

class CerebrumMindHTTPHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Override to suppress console pollution during polling
        pass

    def send_json_response(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def send_static_file(self, filename, content_type):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, "web", filename)
        
        # Security check to prevent directory traversal
        real_base = os.path.realpath(os.path.join(base_dir, "web"))
        real_file = os.path.realpath(file_path)
        if not real_file.startswith(real_base):
            self.send_error(403, "Access Denied")
            return
            
        if not os.path.exists(file_path):
            self.send_error(404, "File Not Found")
            return

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        
        with open(file_path, "rb") as f:
            self.wfile.write(f.read())

    def do_OPTIONS(self):
        # Handle preflight CORS requests
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        # ------------------------------------------------------------- Static Files
        if path in ["", "/", "/index.html"]:
            self.send_static_file("index.html", "text/html; charset=utf-8")
        elif path == "/style.css":
            self.send_static_file("style.css", "text/css")
        elif path == "/app.js":
            self.send_static_file("app.js", "application/javascript")
            
        # ------------------------------------------------------------- API Endpoints
        elif path == "/api/status":
            self.send_json_response(os_manager.get_status())
            
        elif path == "/api/robots":
            self.send_json_response({
                "active": os_manager.active_robot_name,
                "list": list(get_all_robots().values())
            })
            
        elif path == "/api/train/status":
            with os_manager.lock:
                self.send_json_response(os_manager.training_metrics)
                
        elif path == "/api/task/status":
            self.send_json_response(os_manager.get_task_status())
                
        elif path == "/api/advice":
            self.send_json_response(os_manager.get_ai_advice())
            
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        # Read body content
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length) if content_length > 0 else b""
        
        body = {}
        if post_data:
            try:
                body = json.loads(post_data.decode("utf-8"))
            except Exception as e:
                self.send_json_response({"success": False, "error": f"Invalid JSON: {e}"}, 400)
                return

        # ------------------------------------------------------------- API Actions
        if path == "/api/robots/select":
            robot_name = body.get("name")
            if not robot_name:
                self.send_json_response({"success": False, "error": "Robot name parameter missing."}, 400)
                return
            success, msg = os_manager.set_active_robot(robot_name)
            self.send_json_response({"success": success, "message": msg})

        elif path == "/api/robots/custom":
            name = body.get("name")
            joints = body.get("joints", [])
            sensors = body.get("sensors", [])
            height = body.get("height", "1.5 m")
            weight = body.get("weight", "50 kg")
            robot_class = body.get("class", "Custom")
            
            if not name:
                self.send_json_response({"success": False, "error": "Custom robot name is required."}, 400)
                return
            if not joints:
                self.send_json_response({"success": False, "error": "At least one joint must be defined."}, 400)
                return
            
            config = {
                "name": name,
                "class": robot_class,
                "height": height,
                "weight": weight,
                "joints": joints,
                "sensors": sensors
            }
            
            try:
                save_custom_robot(config)
                # Refresh manager list
                os_manager.robots = get_all_robots()
                self.send_json_response({"success": True, "message": f"Custom robot '{name}' successfully registered."})
            except Exception as e:
                self.send_json_response({"success": False, "error": str(e)}, 500)

        elif path == "/api/train/start":
            success, msg = os_manager.start_training()
            self.send_json_response({"success": success, "message": msg})

        elif path == "/api/train/stop":
            success, msg = os_manager.stop_training()
            self.send_json_response({"success": success, "message": msg})

        elif path == "/api/task/run":
            task_name = body.get("task")
            if not task_name:
                self.send_json_response({"success": False, "error": "Task name is required."}, 400)
                return
            success, msg = os_manager.start_task(task_name)
            self.send_json_response({"success": success, "message": msg})

        elif path == "/api/task/stop":
            success, msg = os_manager.stop_task()
            self.send_json_response({"success": success, "message": msg})

        else:
            self.send_error(404, "Not Found")

def run_server(port=8000):
    server_address = ("", port)
    # ThreadingTCPServer handles concurrent requests (important for HTTP polling status checks)
    httpd = ThreadingTCPServer(server_address, CerebrumMindHTTPHandler)
    print(f"Cerebrum-Mind OS server starting on port {port}...")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
        httpd.server_close()

if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    run_server(port)
