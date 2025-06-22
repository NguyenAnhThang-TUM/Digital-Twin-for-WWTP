#!/usr/bin/env python3
"""
SUMO Controller with Automatic SUMO Detection
Finds SUMO installation without relying on Windows registry
"""

import sys
import os
import traceback
import logging
import glob
from pathlib import Path
import subprocess
import time
from ctypes import cdll, c_char_p, c_int

class AutoDetectSumoController:
    def __init__(self, project_file, debug_mode=True):
        self.project_file = project_file
        self.debug_mode = debug_mode
        self.sumo_process = None
        self.running = False
        self.variables = {}
        
        # Setup logging
        self._setup_debug_logging()
        
        # Auto-detect SUMO installation
        self._auto_detect_sumo()
        self._setup_dmq()
        
    def _setup_debug_logging(self):
        """Setup logging that handles encoding issues"""
        log_file = "sumo_debug.log"
        
        # Custom formatter that handles encoding issues
        class SafeFormatter(logging.Formatter):
            def format(self, record):
                try:
                    return super().format(record)
                except UnicodeDecodeError:
                    record.msg = str(record.msg).encode('utf-8', errors='replace').decode('utf-8')
                    return super().format(record)
        
        # Setup handlers
        file_handler = logging.FileHandler(log_file, encoding='utf-8', errors='replace')
        console_handler = logging.StreamHandler(sys.stdout)
        
        formatter = SafeFormatter(
            '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger = logging.getLogger("SumoController")
        self.logger.setLevel(logging.DEBUG if self.debug_mode else logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info("Auto-detect SUMO controller initialized")
        
    def _auto_detect_sumo(self):
        """Automatically detect SUMO installation"""
        self.logger.info("Auto-detecting SUMO installation...")
        
        # Common SUMO installation paths
        potential_paths = [
            r"C:\Program Files\Dynamita\Sumo24",
            r"C:\Program Files\Dynamita\Sumo23", 
            r"C:\Program Files\Dynamita\Sumo22",
            r"C:\Program Files (x86)\Dynamita\Sumo24",
            r"C:\Program Files (x86)\Dynamita\Sumo23",
            r"C:\Program Files (x86)\Dynamita\Sumo22",
            r"C:\Dynamita\Sumo24",
            r"C:\Dynamita\Sumo23",
            r"C:\Dynamita\Sumo22",
        ]
        
        # Also search current directory and parent directories
        current_dir = Path(os.getcwd())
        for parent in [current_dir] + list(current_dir.parents):
            potential_paths.extend([
                str(parent / "Dynamita" / "Sumo24"),
                str(parent / "Dynamita" / "Sumo23"),
                str(parent / "SUMO"),
                str(parent / "sumo"),
            ])
        
        # Search using glob patterns
        search_patterns = [
            r"C:\**\Dynamita\Sumo*",
            r"C:\**\SUMO*",
            r"D:\**\Dynamita\Sumo*",
        ]
        
        for pattern in search_patterns:
            try:
                for path in glob.glob(pattern, recursive=True):
                    if os.path.isdir(path):
                        potential_paths.append(path)
            except:
                pass  # Ignore permission errors
        
        # Test each potential path
        for path in potential_paths:
            if self._test_sumo_path(path):
                self.sumo_path = path
                self.version = os.path.basename(path)
                self.logger.info(f"✓ SUMO found at: {self.sumo_path}")
                self.logger.info(f"✓ Version: {self.version}")
                
                # Try to find license file
                self._find_license_file()
                return
                
        # If nothing found, ask user
        self._manual_sumo_setup()
        
    def _test_sumo_path(self, path):
        """Test if a path contains a valid SUMO installation"""
        if not os.path.exists(path):
            return False
            
        # Check for required files
        required_files = [
            "DMQClient.dll",
            "dmq.exe"
        ]
        
        # Check for SUMO executable (various possible names)
        sumo_exes = [
            "Sumo24.exe",
            "Sumo23.exe", 
            "Sumo22.exe",
            "SUMO.exe",
            "sumo.exe"
        ]
        
        # Test required files
        for file in required_files:
            if not os.path.exists(os.path.join(path, file)):
                self.logger.debug(f"Missing required file in {path}: {file}")
                return False
                
        # Test for at least one SUMO executable
        exe_found = False
        for exe in sumo_exes:
            if os.path.exists(os.path.join(path, exe)):
                self.sumo_exe = exe
                exe_found = True
                break
                
        if not exe_found:
            self.logger.debug(f"No SUMO executable found in {path}")
            return False
            
        # Test PythonAPI directory
        python_api_path = os.path.join(path, "PythonAPI")
        if os.path.exists(python_api_path):
            self.python_api_path = python_api_path
        else:
            self.logger.debug(f"No PythonAPI directory in {path}")
            
        self.logger.debug(f"✓ Valid SUMO installation found: {path}")
        return True
        
    def _find_license_file(self):
        """Find SUMO license file"""
        # Common license file locations
        license_paths = [
            os.path.join(self.sumo_path, "license.lic"),
            os.path.join(self.sumo_path, "sumo.lic"),
            os.path.join(self.sumo_path, "dynamita.lic"),
            os.path.join(os.path.dirname(self.sumo_path), "license.lic"),
        ]
        
        # Search for .lic files in SUMO directory
        for lic_file in glob.glob(os.path.join(self.sumo_path, "*.lic")):
            license_paths.append(lic_file)
            
        for lic_path in license_paths:
            if os.path.exists(lic_path):
                self.license_file = lic_path
                self.logger.info(f"✓ License found: {self.license_file}")
                return
                
        self.logger.warning("⚠ No license file found - DMQ may not work properly")
        self.license_file = ""
        
    def _manual_sumo_setup(self):
        """Manual SUMO setup when auto-detection fails"""
        print("\n" + "="*60)
        print("SUMO AUTO-DETECTION FAILED")
        print("="*60)
        print("Please help locate your SUMO installation:")
        print("")
        
        while True:
            path = input("Enter SUMO installation path (or 'quit' to exit): ").strip()
            
            if path.lower() == 'quit':
                sys.exit(1)
                
            if path.startswith('"') and path.endswith('"'):
                path = path[1:-1]  # Remove quotes
                
            if self._test_sumo_path(path):
                self.sumo_path = path
                self.version = os.path.basename(path)
                self.logger.info(f"✓ Manual SUMO path accepted: {self.sumo_path}")
                self._find_license_file()
                return
            else:
                print(f"✗ Invalid SUMO path: {path}")
                print("Please check that the path contains DMQClient.dll and a SUMO executable")
                
    def _safe_decode(self, byte_data, encodings=['utf-8', 'windows-1252', 'latin-1', 'ascii']):
        """Safely decode bytes with multiple encoding attempts"""
        if isinstance(byte_data, str):
            return byte_data
            
        if byte_data is None:
            return ""
            
        for encoding in encodings:
            try:
                return byte_data.decode(encoding)
            except (UnicodeDecodeError, AttributeError):
                continue
                
        # Fallback
        try:
            return byte_data.decode('utf-8', errors='replace')
        except:
            return str(byte_data)
            
    def _safe_encode(self, text, encoding='utf-8'):
        """Safely encode text to bytes"""
        if isinstance(text, bytes):
            return text
        try:
            return text.encode(encoding)
        except UnicodeEncodeError:
            return text.encode(encoding, errors='replace')
            
    def _setup_dmq(self):
        """Initialize DMQ with proper error handling"""
        try:
            dmq_dll_path = os.path.join(self.sumo_path, "DMQClient.dll")
            
            # Start DMQ if needed
            if not self._is_dmq_running():
                self.logger.info("Starting DMQ server...")
                self._start_dmq_server()
                
            # Load DMQ library
            original_dir = os.getcwd()
            os.chdir(self.sumo_path)
            self.dmq_dll = cdll.LoadLibrary(dmq_dll_path)
            os.chdir(original_dir)
            
            # Setup function signatures
            self.dmq_dll.initModule.argtypes = [c_char_p]
            self.dmq_dll.createQueue.restype = c_char_p
            self.dmq_dll.sendText.argtypes = [c_char_p, c_char_p]
            self.dmq_dll.getText.argtypes = [c_char_p, c_int]
            self.dmq_dll.getText.restype = c_char_p
            self.dmq_dll.closeQueue.argtypes = [c_char_p]
            
            # Initialize DMQ
            self.dmq_dll.initModule("Python".encode("utf-8"))
            queue_bytes = self.dmq_dll.createQueue()
            self.queue_key = self._safe_decode(queue_bytes)
            
            self.logger.info(f"✓ DMQ initialized with key: {self.queue_key}")
            
        except Exception as e:
            self.logger.error(f"DMQ setup failed: {e}")
            raise
            
    def _is_dmq_running(self):
        """Check if DMQ is running"""
        try:
            result = subprocess.run(
                ["tasklist", "/fi", "ImageName eq dmq.exe"], 
                capture_output=True, 
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            return not result.stdout.startswith("INFO:")
        except Exception as e:
            self.logger.warning(f"Could not check DMQ status: {e}")
            return False
            
    def _start_dmq_server(self):
        """Start DMQ server"""
        try:
            os.makedirs("dmq_logs", exist_ok=True)
            
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            batch_file = f"dmq_logs/dmq_{timestamp}.bat"
            
            with open(batch_file, 'w', encoding='utf-8') as f:
                f.write("@echo off\n")
                if self.license_file:
                    f.write(f'"{self.sumo_path}\\DMQ.exe" -autoexit -run "{self.license_file}" >dmq_logs/dmq_{timestamp}.log 2>&1\n')
                else:
                    f.write(f'"{self.sumo_path}\\DMQ.exe" -autoexit >dmq_logs/dmq_{timestamp}.log 2>&1\n')
            
            subprocess.Popen([batch_file], shell=True)
            time.sleep(3)
            
            if self._is_dmq_running():
                self.logger.info("✓ DMQ server started successfully")
            else:
                self.logger.warning("⚠ DMQ may not have started properly")
                
        except Exception as e:
            self.logger.error(f"Error starting DMQ: {e}")
            raise
            
    def start_simulation(self, wait_timeout=60):
        """Start SUMO simulation"""
        try:
            if not os.path.exists(self.project_file):
                raise FileNotFoundError(f"Project file not found: {self.project_file}")
                
            sumo_exe_path = os.path.join(self.sumo_path, self.sumo_exe)
            cmd = [sumo_exe_path, self.project_file, "-dmq", self.queue_key]
            
            self.logger.info(f"Starting SUMO: {' '.join(cmd)}")
            
            self.sumo_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding='utf-8',
                errors='replace'
            )
            
            self.running = True
            
            # Wait for project load
            self.logger.info("Waiting for project to load...")
            if not self._wait_for_message("project_loaded", timeout=wait_timeout):
                # Check if process is still running
                if not self.is_running():
                    stdout, stderr = self.sumo_process.communicate()
                    self.logger.error(f"SUMO process ended. stdout: {stdout}, stderr: {stderr}")
                raise TimeoutError("Project failed to load")
                
            self.logger.info("✓ Project loaded successfully")
            
            # Initialize simulation
            self.send_command("maintab simulate")
            if not self._wait_for_message("model_init", timeout=wait_timeout):
                raise TimeoutError("Model failed to initialize")
                
            self.logger.info("✓ Model initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start simulation: {e}")
            self.stop()
            raise
            
    def send_command(self, command):
        """Send command with encoding safety"""
        try:
            if not self.running:
                raise Exception("Simulation not running")
                
            cmd_bytes = self._safe_encode(command)
            queue_bytes = self._safe_encode(self.queue_key)
            
            result = self.dmq_dll.sendText(queue_bytes, cmd_bytes)
            self.logger.debug(f"Sent command: {command} (result: {result})")
            
        except Exception as e:
            self.logger.error(f"Failed to send command '{command}': {e}")
            raise
            
    def read_message(self, blocking=False):
        """Read message from DMQ with encoding safety"""
        try:
            queue_bytes = self._safe_encode(self.queue_key)
            blocking_int = 1 if blocking else 0
            
            msg_ptr = self.dmq_dll.getText(queue_bytes, blocking_int)
            
            if msg_ptr:
                message = self._safe_decode(msg_ptr)
                return message if message and message.strip() else None
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error reading message: {e}")
            return None
            
    def _wait_for_message(self, expected_msg, timeout=30):
        """Wait for message with encoding safety"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                message = self.read_message(blocking=False)
                
                if message:
                    self.logger.debug(f"SUMO message: {message}")
                    if expected_msg in message:
                        return True
                        
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.warning(f"Error while waiting for message: {e}")
                time.sleep(0.5)
                
        return False
        
    def set_variable(self, var_name, value):
        """Set variable with encoding safety"""
        try:
            command = f"core_cmd set {var_name} {value};"
            self.send_command(command)
            self.variables[var_name] = value
            self.logger.info(f"Set variable: {var_name} = {value}")
            
        except Exception as e:
            self.logger.error(f"Failed to set variable {var_name}: {e}")
            raise
            
    def run_communication_loop(self):
        """Main communication loop"""
        self.logger.info("Starting communication loop...")
        
        try:
            while self.running and self.is_running():
                try:
                    message = self.read_message(blocking=False)
                    
                    if message:
                        self._handle_message(message)
                        
                    time.sleep(0.1)
                    
                except Exception as e:
                    self.logger.warning(f"Communication error: {e}")
                    time.sleep(0.5)
                    
        except KeyboardInterrupt:
            self.logger.info("Stopping (Ctrl+C pressed)...")
        except Exception as e:
            self.logger.error(f"Fatal error in communication loop: {e}")
        finally:
            self.stop()
            
    def _handle_message(self, message):
        """Handle messages safely"""
        try:
            safe_message = str(message).encode('utf-8', errors='replace').decode('utf-8')
            self.logger.debug(f"Handling message: {safe_message}")
            
            if message == "CLOSED":
                self.running = False
                self.logger.info("SUMO connection closed")
                
        except Exception as e:
            self.logger.warning(f"Error handling message: {e}")
            
    def is_running(self):
        """Check if SUMO process is running"""
        return self.sumo_process and self.sumo_process.poll() is None
        
    def stop(self):
        """Stop simulation with cleanup"""
        self.logger.info("Stopping simulation...")
        self.running = False
        
        try:
            if hasattr(self, 'dmq_dll') and hasattr(self, 'queue_key'):
                queue_bytes = self._safe_encode(self.queue_key)
                self.dmq_dll.closeQueue(queue_bytes)
                
            if self.sumo_process and self.is_running():
                self.sumo_process.terminate()
                try:
                    self.sumo_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.sumo_process.kill()
                    
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            
        self.logger.info("✓ Simulation stopped")

def test_auto_detect():
    """Test the auto-detection functionality"""
    print("Testing SUMO Auto-Detection")
    print("="*50)
    
    # You can put any project file path here, or even a non-existent one for testing
    project_file = r"test_project.sumo"  # This doesn't need to exist for basic testing
    
    try:
        controller = AutoDetectSumoController(project_file, debug_mode=True)
        
        print("✓ Controller created successfully")
        print(f"✓ SUMO path: {controller.sumo_path}")
        print(f"✓ SUMO executable: {controller.sumo_exe}")
        print(f"✓ DMQ key: {controller.queue_key}")
        
        # Test variable setting
        controller.set_variable("Test_Variable", 123.45)
        print("✓ Variable setting works")
        
        print("\n✓ Basic functionality test passed!")
        print("\nTo test with actual simulation:")
        print("1. Update project_file path to your actual .sumo file")
        print("2. Uncomment the simulation test lines")
        
        return controller
        
    except Exception as e:
        print(f"✗ Error: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_auto_detect()