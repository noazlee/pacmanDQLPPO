"""
Enhanced DQL Model Evaluation Script
Provides detailed performance metrics and statistics
"""

import subprocess
import time
import sys
import os
import signal
import threading
from pathlib import Path
import json
from datetime import datetime
import statistics


class DQLEvaluator:
    def __init__(self, model_path="fixed_dql_ep10000.pt", num_episodes=10):
        self.model_path = model_path
        self.num_episodes = num_episodes
        self.processes = []
        self.running = True
        self.game_results = []
        self.current_episode = 0
        
    def verify_model_exists(self):
        """Check if the model file exists"""
        if not os.path.exists(self.model_path):
            print(f"❌ Model file '{self.model_path}' not found!")
            print("Available .pt files in current directory:")
            pt_files = list(Path('.').glob('*.pt'))
            if pt_files:
                for f in pt_files:
                    print(f"  - {f}")
                return False
            else:
                print("  No .pt files found")
                return False
        else:
            print(f"✅ Model file '{self.model_path}' found")
            return True
    
    def start_server(self):
        """Start the Pacman server"""
        print("🚀 Starting Pacman server...")
        
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
            
            # Wait for server to start
            time.sleep(3)
            
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
        """Start the visual viewer (optional)"""
        print("🎮 Starting visual viewer...")
        
        cmd = [sys.executable, "viewer.py"]
        
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            self.processes.append(("Viewer", proc))
            
            time.sleep(2)
            
            if proc.poll() is None:
                print("✅ Viewer started successfully")
                return True
            else:
                print("⚠️  Viewer failed to start (continuing without visualization)")
                return False
                
        except Exception as e:
            print(f"⚠️  Error starting viewer: {e}")
            return False
    
    def run_single_episode(self, episode_num):
        """Run a single episode and collect results"""
        print(f"\n🎯 Episode {episode_num + 1}/{self.num_episodes}")
        print("-" * 40)
        
        # Modify the DQL client to use our specific model
        cmd = [sys.executable, "dql_server_client.py", "--model", self.model_path]
        
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            episode_data = {
                'episode': episode_num + 1,
                'steps': 0,
                'final_score': 0,
                'lives_lost': 0,
                'start_time': time.time(),
                'actions_taken': [],
                'score_progression': []
            }
            
            # Monitor agent output
            step_count = 0
            max_steps = 1000  # Prevent infinite games
            
            for line in iter(proc.stdout.readline, ''):
                if not self.running:
                    break
                
                line = line.strip()
                if line:
                    print(f"  {line}")
                    
                    # Parse game information
                    if "Step" in line and "Score:" in line:
                        try:
                            parts = line.split('|')
                            step_part = parts[0].strip()
                            score_part = parts[1].strip()
                            lives_part = parts[2].strip()
                            action_part = parts[3].strip() if len(parts) > 3 else ""
                            
                            step_num = int(step_part.split()[1])
                            score = int(score_part.split()[1])
                            lives = int(lives_part.split()[1])
                            action = action_part.split()[-1] if action_part else ""
                            
                            episode_data['steps'] = step_num
                            episode_data['final_score'] = score
                            episode_data['score_progression'].append((step_num, score))
                            if action:
                                episode_data['actions_taken'].append(action)
                                
                            step_count = step_num
                            
                        except (ValueError, IndexError):
                            pass
                    
                    elif "GAME OVER" in line:
                        try:
                            final_score = int(line.split("Final Score:")[1].strip())
                            episode_data['final_score'] = final_score
                        except (ValueError, IndexError):
                            pass
                        break
                    
                    elif "Lives:" in line:
                        try:
                            lives = int(line.split("Lives:")[1].split()[0])
                            episode_data['lives_lost'] = 3 - lives
                        except (ValueError, IndexError):
                            pass
                
                # Safety check for infinite games
                if step_count > max_steps:
                    print(f"⚠️  Episode exceeded {max_steps} steps, terminating...")
                    proc.terminate()
                    break
            
            episode_data['end_time'] = time.time()
            episode_data['duration'] = episode_data['end_time'] - episode_data['start_time']
            
            # Wait for process to finish
            proc.wait(timeout=5)
            
            return episode_data
            
        except Exception as e:
            print(f"❌ Error in episode {episode_num + 1}: {e}")
            return None
    
    def calculate_statistics(self):
        """Calculate performance statistics"""
        if not self.game_results:
            return {}
        
        scores = [r['final_score'] for r in self.game_results]
        steps = [r['steps'] for r in self.game_results]
        durations = [r['duration'] for r in self.game_results]
        lives_lost = [r['lives_lost'] for r in self.game_results]
        
        stats = {
            'total_episodes': len(self.game_results),
            'score_stats': {
                'mean': statistics.mean(scores),
                'median': statistics.median(scores),
                'min': min(scores),
                'max': max(scores),
                'std_dev': statistics.stdev(scores) if len(scores) > 1 else 0
            },
            'steps_stats': {
                'mean': statistics.mean(steps),
                'median': statistics.median(steps),
                'min': min(steps),
                'max': max(steps)
            },
            'duration_stats': {
                'mean': statistics.mean(durations),
                'total': sum(durations)
            },
            'survival_stats': {
                'avg_lives_lost': statistics.mean(lives_lost),
                'games_completed': sum(1 for r in self.game_results if r['lives_lost'] < 3)
            }
        }
        
        return stats
    
    def print_detailed_results(self):
        """Print comprehensive evaluation results"""
        print("\n" + "="*60)
        print("🏆 DQL MODEL EVALUATION RESULTS")
        print("="*60)
        
        if not self.game_results:
            print("❌ No game results to analyze")
            return
        
        stats = self.calculate_statistics()
        
        print(f"\n📊 PERFORMANCE SUMMARY ({stats['total_episodes']} episodes)")
        print("-" * 40)
        print(f"Model: {self.model_path}")
        print(f"Average Score: {stats['score_stats']['mean']:.1f}")
        print(f"Best Score: {stats['score_stats']['max']}")
        print(f"Worst Score: {stats['score_stats']['min']}")
        print(f"Score Std Dev: {stats['score_stats']['std_dev']:.1f}")
        print(f"Median Score: {stats['score_stats']['median']:.1f}")
        
        print(f"\n🎮 GAMEPLAY METRICS")
        print("-" * 40)
        print(f"Average Steps: {stats['steps_stats']['mean']:.1f}")
        print(f"Longest Game: {stats['steps_stats']['max']} steps")
        print(f"Shortest Game: {stats['steps_stats']['min']} steps")
        print(f"Average Duration: {stats['duration_stats']['mean']:.1f} seconds")
        print(f"Games Completed: {stats['survival_stats']['games_completed']}/{stats['total_episodes']}")
        print(f"Average Lives Lost: {stats['survival_stats']['avg_lives_lost']:.1f}")
        
        print(f"\n📈 INDIVIDUAL GAME RESULTS")
        print("-" * 40)
        for i, result in enumerate(self.game_results, 1):
            status = "✅ Completed" if result['lives_lost'] < 3 else "💀 Game Over"
            print(f"Game {i:2d}: Score {result['final_score']:3d} | Steps {result['steps']:3d} | {status}")
        
        # Performance rating
        avg_score = stats['score_stats']['mean']
        print(f"\n🏅 PERFORMANCE RATING")
        print("-" * 40)
        if avg_score >= 50:
            rating = "🌟 EXCELLENT"
        elif avg_score >= 20:
            rating = "👍 GOOD"
        elif avg_score >= 10:
            rating = "🆗 FAIR"
        elif avg_score >= 5:
            rating = "👎 POOR"
        else:
            rating = "💥 TERRIBLE"
        
        print(f"Rating: {rating} (avg score: {avg_score:.1f})")
        
        # Save results to file
        self.save_results(stats)
    
    def save_results(self, stats):
        """Save evaluation results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_{Path(self.model_path).stem}_{timestamp}.json"
        
        evaluation_data = {
            'model_path': self.model_path,
            'timestamp': timestamp,
            'statistics': stats,
            'individual_games': self.game_results
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(evaluation_data, f, indent=2)
            print(f"\n💾 Results saved to: {filename}")
        except Exception as e:
            print(f"⚠️  Could not save results: {e}")
    
    def cleanup(self):
        """Clean up all processes"""
        print("\n🧹 Cleaning up processes...")
        self.running = False
        
        for name, proc in self.processes:
            if proc.poll() is None:
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
    
    def evaluate(self, show_viewer=False):
        """Run complete evaluation"""
        print("🔍 DQL MODEL EVALUATION STARTING")
        print("="*50)
        print(f"Model: {self.model_path}")
        print(f"Episodes: {self.num_episodes}")
        print(f"Show viewer: {show_viewer}")
        print("="*50)
        
        # Verify model exists
        if not self.verify_model_exists():
            return
        
        try:
            # Start server
            if not self.start_server():
                return
            
            # Optionally start viewer
            if show_viewer:
                self.start_viewer()
            
            # Run evaluation episodes
            print(f"\n🎯 Running {self.num_episodes} evaluation episodes...")
            
            for episode in range(self.num_episodes):
                if not self.running:
                    break
                
                result = self.run_single_episode(episode)
                if result:
                    self.game_results.append(result)
                
                # Brief pause between episodes
                if episode < self.num_episodes - 1:
                    time.sleep(2)
            
            # Display results
            self.print_detailed_results()
                
        except KeyboardInterrupt:
            print("\n⚠️  Evaluation interrupted by user")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
        finally:
            self.cleanup()


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate DQL Pacman Model')
    parser.add_argument('--model', default='fixed_dql_ep10000.pt', 
                       help='Path to model file')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of episodes to run')
    parser.add_argument('--viewer', action='store_true',
                       help='Show visual viewer')
    
    args = parser.parse_args()
    
    evaluator = DQLEvaluator(
        model_path=args.model,
        num_episodes=args.episodes
    )
    
    evaluator.evaluate(show_viewer=args.viewer)


if __name__ == "__main__":
    main()