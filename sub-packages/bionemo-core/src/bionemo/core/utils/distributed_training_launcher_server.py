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

class DistributedTrainingLauncherServer:
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
            
            # Start process
            process = subprocess.Popen(
                cmd,
                cwd=working_dir,
                env=env,
                stdout=open(log_file, 'w'),
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid  # Create new process group
            )
            
            # Store process info
            process_id = self.process_counter
            self.active_processes[process_id] = {
                'process': process,
                'cmd': cmd,
                'job_name': job_name,
                'log_file': str(log_file),
                'start_time': datetime.now().isoformat(),
                'working_dir': working_dir
            }
            
            self.process_counter += 1
            
            # Start monitoring thread
            monitor_thread = threading.Thread(
                target=self.monitor_process,
                args=(process_id,)
            )
            monitor_thread.daemon = True
            monitor_thread.start()
            
            self.logger.info(f"Launched process {process_id}: {' '.join(cmd)}")
            
            return {
                "status": "success",
                "process_id": process_id,
                "job_name": job_name,
                "log_file": str(log_file),
                "pid": process.pid
            }
            
        except Exception as e:
            self.logger.error(f"Error launching process: {e}")
            return {"status": "error", "message": str(e)}
    
    def monitor_process(self, process_id):
        """Monitor a process and clean up when it finishes"""
        if process_id not in self.active_processes:
            return
            
        process_info = self.active_processes[process_id]
        process = process_info['process']
        
        try:
            # Wait for process to complete
            return_code = process.wait()
            
            # Update process info
            process_info['return_code'] = return_code
            process_info['end_time'] = datetime.now().isoformat()
            
            # Close log file
            if process.stdout:
                process.stdout.close()
                
            self.logger.info(f"Process {process_id} finished with return code {return_code}")
            
        except Exception as e:
            self.logger.error(f"Error monitoring process {process_id}: {e}")
            process_info['error'] = str(e)
    
    def get_status(self, request):
        """Get status of a specific process or all processes"""
        process_id = request.get('process_id')
        
        if process_id is not None:
            if process_id not in self.active_processes:
                return {"status": "error", "message": f"Process {process_id} not found"}
            
            info = self.active_processes[process_id].copy()
            process = info.pop('process')
            info['is_running'] = process.poll() is None
            return {"status": "success", "process_info": info}
        else:
            # Return all processes
            all_processes = {}
            for pid, info in self.active_processes.items():
                proc_info = info.copy()
                process = proc_info.pop('process')
                proc_info['is_running'] = process.poll() is None
                all_processes[pid] = proc_info
            
            return {"status": "success", "processes": all_processes}
    
    def kill_process(self, request):
        """Kill a specific process"""
        process_id = request.get('process_id')
        
        if process_id not in self.active_processes:
            return {"status": "error", "message": f"Process {process_id} not found"}
        
        try:
            process = self.active_processes[process_id]['process']
            
            if process.poll() is not None:
                return {"status": "success", "message": "Process already terminated"}
            
            # Kill process group to ensure all children are terminated
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            
            # Wait a bit, then force kill if necessary
            time.sleep(2)
            if process.poll() is None:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            
            self.logger.info(f"Killed process {process_id}")
            return {"status": "success", "message": f"Process {process_id} killed"}
            
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
                'is_running': info['process'].poll() is None,
                'log_file': info['log_file']
            }
            if 'return_code' in info:
                proc_info['return_code'] = info['return_code']
            if 'end_time' in info:
                proc_info['end_time'] = info['end_time']
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
        
        # Kill all active processes
        for process_id, info in self.active_processes.items():
            try:
                process = info['process']
                if process.poll() is None:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except Exception as e:
                self.logger.error(f"Error cleaning up process {process_id}: {e}")
        
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
    
    launcher = DistributedTrainingLauncherServer(args.host, args.port, args.log_dir)
    
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