"""
Client for Training Launcher Server - Use this in Jupyter notebooks
"""
import socket
import json
import time
from typing import Dict, List, Optional, Any

class DistributedTrainingLauncherClient:
    def __init__(self, host='localhost', port=6789):
        self.host = host
        self.port = port
    
    def _send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send a request to the launcher server"""
        try:
            # Create socket connection
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(30)  # 30 second timeout
            sock.connect((self.host, self.port))
            
            # Send request
            message = json.dumps(request)
            sock.send(message.encode('utf-8'))
            
            # Receive response
            response = sock.recv(4096).decode('utf-8')
            sock.close()
            
            return json.loads(response)
            
        except Exception as e:
            return {"status": "error", "message": f"Connection error: {str(e)}"}
    
    def launch_training(self, command: List[str], job_name: str = None, 
                       working_dir: str = None, env_vars: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Launch a training process
        
        Args:
            command: Command and arguments to execute (e.g., ['train_evo2', '--config', 'config.yaml'])
            job_name: Optional name for the job
            working_dir: Working directory for the process
            env_vars: Environment variables to set
        
        Returns:
            Response dict with process_id, job_name, log_file, etc.
        """
        request = {
            'action': 'launch',
            'command': command,
            'job_name': job_name,
            'working_dir': working_dir,
            'env': env_vars or {}
        }
        
        return self._send_request(request)
    
    def get_status(self, process_id: int = None) -> Dict[str, Any]:
        """
        Get status of a process or all processes
        
        Args:
            process_id: Specific process ID, or None for all processes
        """
        request = {'action': 'status'}
        if process_id is not None:
            request['process_id'] = process_id
            
        return self._send_request(request)
    
    def kill_process(self, process_id: int) -> Dict[str, Any]:
        """Kill a specific process"""
        request = {
            'action': 'kill',
            'process_id': process_id
        }
        return self._send_request(request)
    
    def list_processes(self) -> Dict[str, Any]:
        """List all processes"""
        request = {'action': 'list'}
        return self._send_request(request)
    
    def get_logs(self, process_id: int, lines: int = 50) -> Dict[str, Any]:
        """
        Get logs for a process
        
        Args:
            process_id: Process ID
            lines: Number of lines to retrieve from end of log
        """
        request = {
            'action': 'logs',
            'process_id': process_id,
            'lines': lines
        }
        return self._send_request(request)
    
    def wait_for_completion(self, process_id: int, check_interval: int = 30) -> Dict[str, Any]:
        """
        Wait for a process to complete, with periodic status updates
        
        Args:
            process_id: Process ID to wait for
            check_interval: How often to check status (seconds)
        """
        print(f"Waiting for process {process_id} to complete...")
        
        while True:
            status = self.get_status(process_id)
            
            if status['status'] != 'success':
                return status
            
            process_info = status['process_info']
            
            if not process_info['is_running']:
                print(f"Process {process_id} completed!")
                return status
            
            print(f"Process {process_id} still running... (checked at {time.strftime('%H:%M:%S')})")
            time.sleep(check_interval)
    
    def monitor_logs(self, process_id: int, follow: bool = True, refresh_interval: int = 10):
        """
        Monitor logs in real-time (Jupyter-friendly)
        
        Args:
            process_id: Process ID to monitor
            follow: Whether to keep checking for new logs
            refresh_interval: How often to check for new logs (seconds)
        """
        from IPython.display import display, HTML, clear_output
        import time
        
        last_line_count = 0
        
        try:
            while True:
                logs_response = self.get_logs(process_id, lines=1000)
                
                if logs_response['status'] != 'success':
                    print(f"Error getting logs: {logs_response['message']}")
                    break
                
                current_line_count = logs_response['total_lines']
                
                if current_line_count > last_line_count:
                    clear_output(wait=True)
                    print(f"=== Process {process_id} Logs (Last 50 lines) ===")
                    print(logs_response['logs'])
                    print(f"=== Total lines: {current_line_count} ===")
                    last_line_count = current_line_count
                
                # Check if process is still running
                status = self.get_status(process_id)
                if status['status'] == 'success' and not status['process_info']['is_running']:
                    print(f"\nProcess {process_id} has completed!")
                    break
                
                if not follow:
                    break
                    
                time.sleep(refresh_interval)
                
        except KeyboardInterrupt:
            print(f"\nStopped monitoring process {process_id}")

# Convenience functions for Jupyter
def launch_evo2_training(config_path: str, job_name: str = None, **kwargs) -> int:
    """
    Convenience function to launch EVO2 training
    
    Returns process_id for monitoring
    """
    client = LauncherClient()
    
    # Build command
    command = ['train_evo2']
    if config_path:
        command.extend(['--config', config_path])
    
    # Add any additional arguments
    for key, value in kwargs.items():
        command.extend([f'--{key}', str(value)])
    
    response = client.launch_training(
        command=command,
        job_name=job_name or f"evo2_training_{int(time.time())}"
    )
    
    if response['status'] == 'success':
        process_id = response['process_id']
        print(f"Training launched successfully!")
        print(f"Process ID: {process_id}")
        print(f"Job name: {response['job_name']}")
        print(f"Log file: {response['log_file']}")
        print(f"PID: {response['pid']}")
        return process_id
    else:
        print(f"Failed to launch training: {response['message']}")
        return None

def show_all_jobs():
    """Show all running/completed jobs"""
    client = LauncherClient()
    response = client.list_processes()
    
    if response['status'] == 'success':
        processes = response['processes']
        if not processes:
            print("No processes found")
            return
        
        print(f"{'ID':<4} {'Job Name':<20} {'Status':<10} {'Start Time':<20} {'Command'}")
        print("-" * 80)
        
        for proc in processes:
            status = "Running" if proc['is_running'] else "Completed"
            cmd_str = ' '.join(proc['cmd'][:3]) + ('...' if len(proc['cmd']) > 3 else '')
            print(f"{proc['process_id']:<4} {proc['job_name']:<20} {status:<10} {proc['start_time'][:19]:<20} {cmd_str}")
    else:
        print(f"Error: {response['message']}")

def kill_job(process_id: int):
    """Kill a specific job"""
    client = LauncherClient()
    response = client.kill_process(process_id)
    print(response['message'])

def watch_logs(process_id: int):
    """Watch logs for a process in real-time"""
    client = LauncherClient()
    client.monitor_logs(process_id)