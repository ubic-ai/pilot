#!/usr/bin/env python3
"""
UI Dissector Visual Demo
This script demonstrates the visual UI component analysis capabilities.
"""

import os
import json
import argparse
import webbrowser
import subprocess
import platform
import http.server
import socketserver
import threading
from pathlib import Path
from dotenv import load_dotenv
from ui_dissector import UIDissector

# Load environment variables from .env file
load_dotenv()

def is_running_in_container():
    """Detect if running in a container environment"""
    # Check for container-specific files or environment variables
    container_indicators = [
        os.path.exists('/.dockerenv'),                     # Docker
        os.path.exists('/run/.containerenv'),              # Podman/other OCI
        'REMOTE_CONTAINERS' in os.environ,                 # VS Code Remote Containers
        'CODESPACES' in os.environ,                        # GitHub Codespaces
        'CONTAINER' in os.environ,                         # Generic container flag
        'KUBERNETES_SERVICE_HOST' in os.environ,           # Kubernetes
        os.path.exists('/var/run/secrets/kubernetes.io'),  # Kubernetes
    ]
    
    return any(container_indicators)

def serve_html_report(directory, port=8000):
    """Serve the HTML report on a local web server"""
    os.chdir(directory)
    
    handler = http.server.SimpleHTTPRequestHandler
    
    try:
        # Find an available port
        while True:
            try:
                httpd = socketserver.TCPServer(("", port), handler)
                break
            except OSError:
                print(f"Port {port} is busy, trying port {port+1}")
                port += 1
        
        print(f"\nStarting HTTP server on port {port}")
        print(f"Open http://localhost:{port}/report.html in your browser")
        print("Press Ctrl+C to stop the server when done")
        
        # Run the server in a thread so the script can continue
        server_thread = threading.Thread(target=httpd.serve_forever)
        server_thread.daemon = True  # Thread will exit when main thread exits
        server_thread.start()
        
        # Wait for user to press Enter
        input("\nPress Enter to stop the server and exit...\n")
        
        # Shutdown server
        httpd.shutdown()
        print("Server stopped")
        
    except KeyboardInterrupt:
        print("\nServer stopped")
        if 'httpd' in locals():
            httpd.shutdown()

def main():
    parser = argparse.ArgumentParser(description='UI Dissector Visual Demo')
    parser.add_argument('image_path', help='Path to design image')
    parser.add_argument('-o', '--output-dir', help='Output directory for results and visualizations')
    parser.add_argument('-a', '--api-key', help='Roboflow API key (can also use ROBOFLOW_API_KEY env var)')
    parser.add_argument('--api-url', help='Custom Roboflow API URL')
    parser.add_argument('-t', '--threshold', type=float, default=0.4, help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--no-browser', action='store_true', help='Do not open results in browser')
    parser.add_argument('--force-simulated', action='store_true', help='Force using simulated detection (no API)')
    parser.add_argument('--port', type=int, default=8000, help='Port to use for web server in container mode')
    parser.add_argument('--container-mode', action='store_true', help='Force container mode (use web server)')
    args = parser.parse_args()
    
    # Get API key from args or environment
    api_key = args.api_key or os.environ.get("ROBOFLOW_API_KEY", "")
    
    if args.force_simulated:
        api_key = ""
        print("Forcing simulated detection mode (no API call)")
    
    # Check for API key
    if not api_key:
        print("No API key provided. Using simulated detection.")
        print("To use the Roboflow API, provide an API key with --api-key or set ROBOFLOW_API_KEY environment variable.")
        print("See ROBOFLOW_SETUP.md for instructions on setting up a Roboflow API key.")
    
    # Configure the UI Dissector
    config = {
        'api_key': api_key,
        'confidence_threshold': args.threshold,
    }
    
    # Add custom API URL if provided
    if args.api_url:
        config['api_url'] = args.api_url
    
    # Initialize UI Dissector
    dissector = UIDissector(config)
    
    # Process design file
    print(f"Analyzing design file: {args.image_path}")
    try:
        result = dissector.process_design(
            image_path=args.image_path,
            output_dir=args.output_dir,
            visualize=True
        )
        
        if not result:
            print("Error: Failed to analyze design file")
            return
        
        # Print analysis completion message
        print("\n===== UI DISSECTOR ANALYSIS COMPLETE =====")
        print(f"Total components detected: {result['metadata']['component_count']}")
        print(f"Processing time: {result['metadata']['processing_time']:.2f} seconds")
        print(f"Results saved to: {result['metadata']['output_directory']}")
        
        # Open HTML report in browser if available
        if 'visualizations' in result and 'html_report' in result['visualizations']:
            html_path = result['visualizations']['html_report']
            print(f"HTML report generated: {html_path}")
            
            # Detect if running in container or container mode is forced
            in_container = is_running_in_container() or args.container_mode
            
            if not args.no_browser:
                if in_container:
                    print("Running in container environment - starting local web server")
                    # Start a simple HTTP server to view the results
                    serve_html_report(result['metadata']['output_directory'], args.port)
                else:
                    print("Opening report in web browser...")
                    html_path_abs = os.path.abspath(html_path)
                    html_url = f"file://{html_path_abs}"
                    print(f"Opening URL: {html_url}")
                    
                    # Try to open browser with multiple methods
                    browser_success = False
                    
                    try:
                        # Method 1: Standard webbrowser.open
                        browser_success = webbrowser.open(html_url)
                    except Exception as e:
                        print(f"Error opening browser (method 1): {str(e)}")
                    
                    # If first method failed, try with a specific browser
                    if not browser_success:
                        try:
                            # Method 2: Try to get a specific browser controller
                            browser = webbrowser.get()
                            browser.open(html_url)
                            browser_success = True
                        except Exception as e:
                            print(f"Error opening browser (method 2): {str(e)}")
                    
                    # If both methods failed, try with os-specific commands
                    if not browser_success:
                        try:
                            # Method 3: OS-specific commands
                            system = platform.system()
                            if system == 'Darwin':  # macOS
                                subprocess.run(['open', html_path_abs])
                            elif system == 'Windows':
                                subprocess.run(['start', html_path_abs], shell=True)
                            elif system == 'Linux':
                                subprocess.run(['xdg-open', html_path_abs])
                            
                            browser_success = True
                        except Exception as e:
                            print(f"Error opening browser (method 3): {str(e)}")
                    
                    if not browser_success:
                        print("Could not open browser automatically.")
                        print(f"Please open the HTML report manually: {html_path_abs}")
                        print("Or use --container-mode to start a web server")
        
        print("\nSummary by atomic design level:")
        for level, count in result['summary'].items():
            print(f"  {level.capitalize()}: {count}")
        
        print("\n==========================================")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Try running with --force-simulated to use simulated detection instead of the API.")
        return 1

if __name__ == "__main__":
    main()