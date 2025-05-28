#!/usr/bin/env python3
"""
Convenience script to run the full DQL game setup
"""

import subprocess
import time
import sys
import os
import signal
import threading
from pathlib import Path


class GameRunner:
    def __init__(self):
        self.processes = []
        self.running = True
    
    def start_server(self):
        """Start the Pacman server"""
        print("🖥️  Starting Pacman server...")
        
        cmd = [
            sys.executable, "server.py",
            "--map", "data/fixed_classic.bmp",
            "--ghosts", "2",
            "--level", "1"
        ]
        
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.processes.append(("Server", proc))
            
            # Wait a moment for server to start
            time.sleep(2)
            
            if proc.poll() is None:
                print("✅ Server started successfully")
                return True
            else:
                stdout, stderr = proc.communicate()
                print(f"❌ Server failed to start:")
                print(f"STDOUT: {stdout}")
                print(f"STDERR: {stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error starting server: {e}")
            return False
    
    def start_viewer(self):
        """Start the visual viewer"""
        print("👁️  Starting visual viewer...")
        
        cmd = [sys.executable, "viewer.py"]
        
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.processes.append(("Viewer", proc))
            
            # Give viewer time to connect
            time.sleep(1)
            
            if proc.poll() is None:
                print("✅ Viewer started successfully")
                return True
            else:
                stdout, stderr = proc.communicate()
                print(f"⚠️  Viewer may have issues:")
                print(f"STDOUT: {stdout}")
                print(f"STDERR: {stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error starting viewer: {e}")
            return False
    
    def start_dql_agent(self, model_path="pacman_dqn_final.pt"):
        """Start the DQL agent"""
        print("🤖 Starting DQL agent...")
        
        if not os.path.exists(model_path):
            print(f"⚠️  Model {model_path} not found - agent will play randomly")
        
        cmd = [sys.executable, "dql_server_client.py"]
        
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            self.processes.append(("DQL Agent", proc))
            
            # Monitor agent output in real-time
            def monitor_agent():
                for line in iter(proc.stdout.readline, ''):
                    if self.running:
                        print(f"🤖 {line.strip()}")
                    else:
                        break
            
            agent_thread = threading.Thread(target=monitor_agent)
            agent_thread.daemon = True
            agent_thread.start()
            
            print("✅ DQL agent started")
            return True
            
        except Exception as e:
            print(f"❌ Error starting DQL agent: {e}")
            return False
    
    def cleanup(self):
        """Clean up all processes"""
        print("\n🧹 Cleaning up processes...")
        self.running = False
        
        for name, proc in self.processes:
            if proc.poll() is None:  # Process is still running
                print(f"   Stopping {name}...")
                try:
                    proc.terminate()
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    print(f"   Force killing {name}...")
                    proc.kill()
                except Exception as e:
                    print(f"   Error stopping {name}: {e}")
        
        print("✅ Cleanup complete")
    
    def run_full_game(self):
        """Run the complete game setup"""
        print("🎮 Starting Complete DQL Pacman Game")
        print("=" * 50)
        
        try:
            # Check required files
            required_files = ["server.py", "viewer.py", "dql_server_client.py", "data/fixed_classic.bmp"]
            missing_files = [f for f in required_files if not os.path.exists(f)]
            
            if missing_files:
                print(f"❌ Missing required files: {missing_files}")
                return
            
            # Start components in order
            if not self.start_server():
                return
            
            if not self.start_viewer():
                print("⚠️  Continuing without viewer...")
            
            if not self.start_dql_agent():
                return
            
            print("\n🎮 Game is running!")
            print("=" * 50)
            print("You should see:")
            print("  1. Pygame window showing the game")
            print("  2. DQL agent playing automatically") 
            print("  3. Real-time score and progress updates")
            print()
            print("Press Ctrl+C to stop all components")
            print("=" * 50)
            
            # Wait for user interrupt or processes to finish
            try:
                while self.running:
                    # Check if any critical process died
                    for name, proc in self.processes:
                        if proc.poll() is not None and name in ["Server", "DQL Agent"]:
                            print(f"💀 {name} process died")
                            self.running = False
                            break
                    
                    if self.running:
                        time.sleep(1)
                        
            except KeyboardInterrupt:
                print("\n🛑 Interrupted by user")
                
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
        finally:
            self.cleanup()


def main():
    """Main function"""
    if len(sys.argv) > 1:
        if sys.argv[1] == "server-only":
            # Just start server for manual testing
            runner = GameRunner()
            runner.start_server()
            print("Server running. Start viewer and agent manually.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                runner.cleanup()
            return
        elif sys.argv[1] == "help":
            print("Usage:")
            print("  python run_dql_game.py           # Run complete setup")
            print("  python run_dql_game.py server-only  # Just start server")
            return
    
    # Run complete game
    runner = GameRunner()
    runner.run_full_game()


if __name__ == "__main__":
    main()