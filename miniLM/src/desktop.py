"""
MiniChat Desktop Application
Wraps the Streamlit app in a native desktop window using PyWebView.
"""

import subprocess
import sys
import time
import threading
import socket
import webview
import signal
import os

# Configuration
STREAMLIT_PORT = 8501
APP_TITLE = "MiniChat"
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800


def find_free_port(start_port: int = 8501) -> int:
    """Find a free port starting from start_port."""
    port = start_port
    while port < start_port + 100:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            port += 1
    return start_port


def wait_for_server(port: int, timeout: int = 30) -> bool:
    """Wait for the Streamlit server to start."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                s.connect(('localhost', port))
                return True
        except (socket.error, socket.timeout):
            time.sleep(0.5)
    return False


class StreamlitServer:
    """Manages the Streamlit server process."""
    
    def __init__(self, port: int):
        self.port = port
        self.process = None
        
    def start(self) -> bool:
        """Start the Streamlit server."""
        # Get the path to app.py
        app_path = os.path.join(os.path.dirname(__file__), 'app.py')
        
        # Build the command
        cmd = [
            sys.executable, '-m', 'streamlit', 'run',
            app_path,
            '--server.port', str(self.port),
            '--server.headless', 'true',
            '--server.address', 'localhost',
            '--browser.gatherUsageStats', 'false',
            '--global.developmentMode', 'false',
        ]
        
        # Start the process
        # Use CREATE_NO_WINDOW on Windows to hide the console
        kwargs = {}
        if sys.platform == 'win32':
            kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
        
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **kwargs
        )
        
        # Wait for server to be ready
        return wait_for_server(self.port)
    
    def stop(self):
        """Stop the Streamlit server."""
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None


def on_closing(window):
    """Handle window close event."""
    window.destroy()


def main():
    """Main entry point for the desktop application."""
    # Find a free port
    port = find_free_port(STREAMLIT_PORT)
    
    # Create and start the Streamlit server
    server = StreamlitServer(port)
    
    print(f"Starting MiniChat on port {port}...")
    
    if not server.start():
        print("Failed to start Streamlit server. Please check if all dependencies are installed.")
        sys.exit(1)
    
    print("Server started. Opening window...")
    
    # Create the webview window
    window = webview.create_window(
        APP_TITLE,
        f'http://localhost:{port}',
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
        resizable=True,
        min_size=(800, 600),
    )
    
    # Handle cleanup on window close
    def cleanup():
        server.stop()
    
    # Start the webview (this blocks until window is closed)
    webview.start()
    
    # Cleanup after window closes
    cleanup()
    print("MiniChat closed.")


if __name__ == '__main__':
    main()
