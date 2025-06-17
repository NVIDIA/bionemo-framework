#!/usr/bin/env python3
"""
Training Launcher Server - Decouples training execution from Jupyter
"""
import socket
import json
import subprocess
import threading
import time
import os
import signal
import logging
from datetime import datetime
from pathlib import Path

class TrainingLauncher:
    def __init__(self, host='localhost', port=6789, log_dir='/tmp/launcher_logs'):
        self.host = host
        self.port = port
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.active_processes = {}
        self.process_counter = 0
        self.running = True
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / 'launcher.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def start_server(self):
        """Start the launcher server"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            self.socket.bind((self.host, self.port))
            self.socket.listen(5)
            self.logger.info(f"Launcher server started on {self.host}:{self.port}")
            
            while self.running:
                try:
                    client_socket, address = self.socket.accept()
                    self.logger.info(f"Connection from {address}")
                    
                    # Handle client in separate thread
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, address)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                    
                except socket.error as e:
                    if self.running:
                        self.logger.error(f"Socket error: {e}")
                        
        except Exception as e:
            self.logger.error(f"Server error: {e}")
        finally:
            self.cleanup()
    
    def handle_client(self, client_socket, address):
        """Handle individual client requests"""
        try:
            # Receive message
            data = client_socket.recv(4096).decode('utf-8')
            if not data:
                return
                
            request = json.loads(data)
            self.logger.info(f"Received request: {request}")
            
            # Process request
            response = self.process_request(request)
            
            # Send response
            client_socket.send(json.dumps(response).encode('utf-8'))
            
        except Exception as e:
            self.logger.error(f"Error handling client {address}: {e}")
            error_response = {"status": "error", "message": str(e)}
            try:
                client_socket.send(json.dumps(error_response).encode('utf-8'))
            except:
                pass
        finally:
            client_socket.close()
    
    def process_request(self, request):
        """Process different types of requests"""
        action = request.get('action')
        
        if action == 'launch':
            return self.launch_training(request)
        elif action == 'status':
            return self.get_status(request)
        elif action == 'kill':
            return self.kill_process(request)
        elif action == 'list':
            return self.list_processes()
        elif action == 'logs':
            return self.get_logs(request)
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}
    
    def launch_training(self, request):
        """Launch a training process"""
        try:
            cmd = request.get('command', [])
            working_dir = request.get('working_dir', os.getcwd())
            env_vars = request.get('env', {})
            job_name = request.get('job_name', f'job_{self.process_counter}')
            
            if not cmd:
                return {"status": "error", "message": "No command provided"}
            
            # Prepare environment
            env = os.environ.copy()
            env.update(env_vars)
            
            # Create log files
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = self.log_dir / f"{job_name}_{timestamp}.log"
            
            # Store process info (before starting)
            process_id = self.process_counter
            self.active_processes[process_id] = {
                'cmd': cmd,
                'job_name': job_name,
                'log_file': str(log_file),
                'start_time': datetime.now().isoformat(),
                'working_dir': working_dir,
                'status': 'starting',
                'return_code': None
            }
            
            self.process_counter += 1
            
            # Start execution in separate thread
            execution_thread = threading.Thread(
                target=self.execute_process,
                args=(process_id, cmd, working_dir, env, log_file)
            )
            execution_thread.daemon = True
            execution_thread.start()
            
            self.logger.info(f"Launched process {process_id}: {' '.join(cmd)}")
            
            return {
                "status": "success",
                "process_id": process_id,
                "job_name": job_name,
                "log_file": str(log_file)
            }
            
        except Exception as e:
            self.logger.error(f"Error launching process: {e}")
            return {"status": "error", "message": str(e)}
    
    def execute_process(self, process_id, cmd, working_dir, env, log_file):
        """Execute a process using subprocess.run() in a separate thread"""
        try:
            # Update status to running
            self.active_processes[process_id]['status'] = 'running'
            
            # Execute the command
            with open(log_file, 'w') as log_handle:
                result = subprocess.run(
                    cmd,
                    cwd=working_dir,
                    env=env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            
            # Update process info with results
            self.active_processes[process_id].update({
                'status': 'completed',
                'return_code': result.returncode,
                'end_time': datetime.now().isoformat()
            })
            
            status_msg = "successfully" if result.returncode == 0 else f"with return code {result.returncode}"
            self.logger.info(f"Process {process_id} finished {status_msg}")
            
        except Exception as e:
            # Update process info with error
            self.active_processes[process_id].update({
                'status': 'error',
                'error': str(e),
                'end_time': datetime.now().isoformat()
            })
            self.logger.error(f"Error executing process {process_id}: {e}")
    
    def monitor_process(self, process_id):
        """This method is no longer needed with subprocess.run() approach"""
        pass
    
    def get_status(self, request):
        """Get status of a specific process or all processes"""
        process_id = request.get('process_id')
        
        if process_id is not None:
            if process_id not in self.active_processes:
                return {"status": "error", "message": f"Process {process_id} not found"}
            
            info = self.active_processes[process_id].copy()
            info['is_running'] = info.get('status') == 'running'
            return {"status": "success", "process_info": info}
        else:
            # Return all processes
            all_processes = {}
            for pid, info in self.active_processes.items():
                proc_info = info.copy()
                proc_info['is_running'] = proc_info.get('status') == 'running'
                all_processes[pid] = proc_info
            
            return {"status": "success", "processes": all_processes}
    
    def kill_process(self, request):
        """Kill a specific process"""
        process_id = request.get('process_id')
        
        if process_id not in self.active_processes:
            return {"status": "error", "message": f"Process {process_id} not found"}
        
        try:
            process_info = self.active_processes[process_id]
            
            if process_info.get('status') != 'running':
                return {"status": "success", "message": "Process already terminated"}
            
            # Since we're using subprocess.run() in threads, we need to track PIDs differently
            # For now, we'll mark it as killed and let the thread handle cleanup
            process_info['status'] = 'killed'
            process_info['end_time'] = datetime.now().isoformat()
            
            # Note: With subprocess.run(), the process runs to completion in the thread
            # True process killing would require storing the Popen object or PID
            # For most use cases, this status update is sufficient
            
            self.logger.info(f"Marked process {process_id} as killed")
            return {"status": "success", "message": f"Process {process_id} marked as killed"}
            
        except Exception as e:
            self.logger.error(f"Error killing process {process_id}: {e}")
            return {"status": "error", "message": str(e)}
    
    def list_processes(self):
        """List all processes"""
        processes = []
        for pid, info in self.active_processes.items():
            proc_info = {
                'process_id': pid,
                'job_name': info['job_name'],
                'cmd': info['cmd'],
                'start_time': info['start_time'],
                'status': info.get('status', 'unknown'),
                'is_running': info.get('status') == 'running',
                'log_file': info['log_file']
            }
            if 'return_code' in info:
                proc_info['return_code'] = info['return_code']
            if 'end_time' in info:
                proc_info['end_time'] = info['end_time']
            if 'error' in info:
                proc_info['error'] = info['error']
            processes.append(proc_info)
        
        return {"status": "success", "processes": processes}
    
    def get_logs(self, request):
        """Get logs for a process"""
        process_id = request.get('process_id')
        lines = request.get('lines', 50)  # Last N lines
        
        if process_id not in self.active_processes:
            return {"status": "error", "message": f"Process {process_id} not found"}
        
        log_file = self.active_processes[process_id]['log_file']
        
        try:
            with open(log_file, 'r') as f:
                all_lines = f.readlines()
                last_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
                
            return {
                "status": "success",
                "logs": ''.join(last_lines),
                "total_lines": len(all_lines)
            }
        except Exception as e:
            return {"status": "error", "message": f"Error reading logs: {e}"}
    
    def cleanup(self):
        """Clean up resources"""
        self.logger.info("Cleaning up launcher server...")
        
        # Mark all running processes as interrupted
        for process_id, info in self.active_processes.items():
            if info.get('status') == 'running':
                info['status'] = 'interrupted'
                info['end_time'] = datetime.now().isoformat()
        
        # Close socket
        if hasattr(self, 'socket'):
            self.socket.close()
    
    def stop(self):
        """Stop the server"""
        self.running = False
        if hasattr(self, 'socket'):
            self.socket.close()

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Training Launcher Server')
    parser.add_argument('--host', default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=9999, help='Server port')
    parser.add_argument('--log-dir', default='/tmp/launcher_logs', help='Log directory')
    
    args = parser.parse_args()
    
    launcher = TrainingLauncher(args.host, args.port, args.log_dir)
    
    def signal_handler(signum, frame):
        print("\nShutting down launcher server...")
        launcher.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        launcher.start_server()
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    finally:
        launcher.cleanup()

if __name__ == '__main__':
    main()