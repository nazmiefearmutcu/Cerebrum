#!/usr/bin/env python3
"""Launch script for Cerebrum-Mind Robot Operating System & Dashboard."""
import sys
import os
import subprocess
import webbrowser

def main():
    print(r"""
======================================================================
  ______ ______ _____  ______ ____  _____  _    _ __  __      __  __ _____ _   _ _____  
 / _____|  ____|  __ \|  ____|  _ \|  __ \| |  | |  \/  |    |  \/  |_   _| \ | |  __ \ 
| |     | |__  | |__) | |__  | |_) | |__) | |  | | \  / |____| \  / | | | |  \| | |  | |
| |     |  __| |  _  /|  __| |  _ <|  _  /| |  | | |\/| |____| |\/| | | | | . ` | |  | |
| |_____| |____| | \ \| |____| |_) | | \ \| |__| | |  | |    | |  | |_| |_| |\  | |__| |
 \______|______|_|  \_\______|____/|_|  \_\\____/|_|  |_|    |_|  |_|_____|_| \_|_____/ 
======================================================================
                  Robot Cognitive Operating System v1.0.0
       Powered by the Backprop-Free Predictive Coding Cerebrum Engine
======================================================================
""")
    
    # Port configuration
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
            
    # Add project root to python path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    
    # Command to run server
    cmd = [sys.executable, "-m", "cerebrum_mind.server", str(port)]
    
    print(f"[*] Starting Cerebrum-Mind OS core engine on port {port}...")
    print(f"[*] Serving web dashboard from: http://localhost:{port}")
    print("[*] Launching web browser dashboard...")
    
    # Auto open browser after a tiny delay
    try:
        webbrowser.open(f"http://localhost:{port}")
    except Exception:
        print("[!] Web browser auto-open failed. Please open http://localhost:8000 manually.")
        
    try:
        subprocess.run(cmd, cwd=project_root, check=True)
    except KeyboardInterrupt:
        print("\n[!] Shutting down Cerebrum-Mind OS kernel.")
    except Exception as e:
        print(f"\n[ERROR] Core crashed: {e}")

if __name__ == "__main__":
    main()
