"""
Parallel Enhanced Self-Play Script with Strategy Experience Pool
Multi-threaded batch execution of EnhancedMichael agents for faster learning.
"""

import textarena as ta
from src.enhanced_agent import EnhancedMichael
from src.strategy_pool import StrategyExperiencePool
from src.game_logger import GameLogger, LoggedAgent
import threading
import queue
import time
import os
import json
from collections import defaultdict, Counter
from datetime import datetime
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Configuration
MODEL_NAME = "EnhancedMichael-Parallel"
ENV_ID = "SecretMafia-v0"
ENABLE_LOGGING = True
LOG_DIR = "parallel_game_logs"
STRATEGY_POOL_FILE = "parallel_strategy_pool.json"

# Agent configurations
AGENT_CONFIGS = {
    0: {"name": "Parallel-Michael-0", "model": "deepseek-v3.1", "enable_learning": True},
    1: {"name": "Parallel-Michael-1", "model": "deepseek-v3.1", "enable_learning": True},
    2: {"name": "Parallel-Michael-2", "model": "deepseek-v3.1", "enable_learning": True},
    3: {"name": "Parallel-Michael-3", "model": "deepseek-v3.1", "enable_learning": True},
    4: {"name": "Parallel-Michael-4", "model": "deepseek-v3.1", "enable_learning": True},
    5: {"name": "Parallel-Michael-5", "model": "deepseek-v3.1", "enable_learning": True},
}

# Thread-safe statistics
stats_lock = threading.Lock()
parallel_stats = {
    "total_games": 0,
    "completed_games": 0,
    "failed_games": 0,
    "mafia_wins": 0,
    "villager_wins": 0,
    "role_stats": defaultdict(lambda: {"wins": 0, "games": 0, "win_rate": 0.0}),
    "agent_stats": defaultdict(lambda: {
        "wins": 0, "games": 0, "win_rate": 0.0,
        "roles": Counter(), "experiences_learned": 0
    }),
    "strategy_pool_stats": {
        "total_experiences": 0,
        "learning_events": []
    },
    "thread_stats": defaultdict(int)
}

class GameWorker(threading.Thread):
    """Worker thread for playing games"""

    def __init__(self, worker_id, game_queue, result_queue, shared_pool):
        super().__init__()
        self.worker_id = worker_id
        self.game_queue = game_queue
        self.result_queue = result_queue
        self.shared_pool = shared_pool
        self.daemon = True

    def run(self):
        """Main worker loop"""
        while True:
            try:
                # Get game task from queue
                game_task = self.game_queue.get(timeout=5)
                if game_task is None:  # Poison pill - stop worker
                    break

                game_id, game_config = game_task
                self.play_game(game_id, game_config)

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker {self.worker_id} error: {e}")
                with stats_lock:
                    parallel_stats["failed_games"] += 1
                    parallel_stats["thread_stats"][f"worker_{self.worker_id}_errors"] += 1
            finally:
                self.game_queue.task_done()

    def play_game(self, game_id, game_config):
        """Play a single game"""
        try:
            # Initialize agents for this game
            agents = {}
            for player_id, config in AGENT_CONFIGS.items():
                agents[player_id] = EnhancedMichael(
                    model_name=config['model'],
                    enable_learning=config['enable_learning']
                )

            # Track initial experience counts
            initial_experience_counts = {}
            for player_id, agent in agents.items():
                if hasattr(agent, 'get_learning_statistics'):
                    initial_experience_counts[player_id] = agent.get_learning_statistics().get('total_experiences', 0)

            # Initialize environment
            env = ta.make(env_id=ENV_ID)
            env.reset(num_players=len(agents))

            # Game loop
            done = False
            turn_count = 0
            learning_events = []

            while not done:
                turn_count += 1
                player_id, observation = env.get_observation()

                # Track learning
                agent = agents[player_id]
                experiences_before = agent.get_learning_statistics()['total_experiences'] if hasattr(agent, 'get_learning_statistics') else 0

                action = agent(observation)

                experiences_after = agent.get_learning_statistics()['total_experiences'] if hasattr(agent, 'get_learning_statistics') else 0
                if experiences_after > experiences_before:
                    learning_events.append({
                        'turn': turn_count,
                        'player_id': player_id,
                        'new_experiences': experiences_after - experiences_before
                    })

                done, step_info = env.step(action=action)

            rewards, game_info = env.close()

            # Calculate final experience counts
            final_experience_counts = {}
            for player_id, agent in agents.items():
                if hasattr(agent, 'get_learning_statistics'):
                    final_count = agent.get_learning_statistics().get('total_experiences', 0)
                    initial_count = initial_experience_counts.get(player_id, 0)
                    final_experience_counts[player_id] = final_count - initial_count

            # Send results to main thread
            self.result_queue.put({
                'game_id': game_id,
                'worker_id': self.worker_id,
                'rewards': rewards,
                'game_info': game_info,
                'turn_count': turn_count,
                'learning_events': learning_events,
                'experience_counts': final_experience_counts,
                'success': True
            })

            with stats_lock:
                parallel_stats["thread_stats"][f"worker_{self.worker_id}_games"] += 1

        except Exception as e:
            print(f"Worker {self.worker_id} game {game_id} failed: {e}")
            self.result_queue.put({
                'game_id': game_id,
                'worker_id': self.worker_id,
                'error': str(e),
                'success': False
            })

def setup_shared_strategy_pool():
    """Setup shared strategy pool with thread safety"""
    pool = StrategyExperiencePool(STRATEGY_POOL_FILE)

    # Add initial experiences only if pool is empty
    if pool.get_statistics()["total_experiences"] == 0:
        print("Adding initial brilliant experiences to strategy pool...")
        add_initial_experiences(pool)

    return pool

def add_initial_experiences(pool):
    """Add initial experiences to jumpstart learning"""
    initial_experiences = [
        {
            "role": "Doctor",
            "phase": "night",
            "round_number": 3,
            "key_info": "Player claims detective role and provides credible evidence",
            "action": "[1]",
            "action_type": "protect",
            "reasoning": "Protect claimed detective as they are valuable to villagers",
            "situation_assessment": "protecting_key_player",
            "strategic_thinking": "Detective role is crucial for identifying Mafia",
            "key_factors": ["role claim", "credible evidence", "protection priority"],
            "success": True,
            "impact": "Successfully protected detective from Mafia attack"
        },
        {
            "role": "Mafia",
            "phase": "day_speak",
            "round_number": 4,
            "key_info": "Being accused, need to deflect suspicion",
            "action": "I think Player 3 has been very quiet and suspicious. As a regular villager, I'm just trying to find the real Mafia.",
            "action_type": "speak",
            "reasoning": "Deflect accusation by pointing suspicion elsewhere",
            "situation_assessment": "deception_opportunity",
            "strategic_thinking": "Create confusion and shift focus to another player",
            "key_factors": ["accusation defense", "deflection strategy", "blame shifting"],
            "success": True,
            "impact": "Successfully diverted votes to eliminate a villager"
        },
        {
            "role": "Detective",
            "phase": "night",
            "round_number": 2,
            "key_info": "Player showing inconsistent behavior and suspicious voting patterns",
            "action": "[4]",
            "action_type": "investigate",
            "reasoning": "Investigate most suspicious player to gather information",
            "situation_assessment": "investigating_suspects",
            "strategic_thinking": "Early investigation helps identify Mafia for elimination",
            "key_factors": ["suspicious behavior", "inconsistent statements", "investigation priority"],
            "success": True,
            "impact": "Discovered Mafia member, shared information with villagers"
        }
    ]

    for exp in initial_experiences:
        pool.add_experience(**exp)

    print(f"Added {len(initial_experiences)} initial experiences to strategy pool")

def update_statistics(game_result):
    """Thread-safe statistics update"""
    with stats_lock:
        if not game_result['success']:
            parallel_stats["failed_games"] += 1
            return

        parallel_stats["completed_games"] += 1
        parallel_stats["total_games"] += 1

        rewards = game_result['rewards']
        game_info = game_result['game_info']
        learning_events = game_result['learning_events']
        experience_counts = game_result['experience_counts']

        # Count wins by team
        mafia_win = any(info["role"] in ["Mafia"] and reward == 1
                       for info, reward in zip(game_info.values(), rewards.values()))

        if mafia_win:
            parallel_stats["mafia_wins"] += 1
        else:
            parallel_stats["villager_wins"] += 1

        # Update role and agent statistics
        for player_id, info in game_info.items():
            role = info["role"]
            reward = rewards[player_id]
            agent_config = AGENT_CONFIGS.get(player_id, {"name": f"agent_{player_id}"})
            agent_name = agent_config["name"]

            # Role stats
            parallel_stats["role_stats"][role]["games"] += 1
            if reward == 1:
                parallel_stats["role_stats"][role]["wins"] += 1

            # Agent stats
            parallel_stats["agent_stats"][agent_name]["games"] += 1
            parallel_stats["agent_stats"][agent_name]["roles"][role] += 1
            parallel_stats["agent_stats"][agent_name]["experiences_learned"] += experience_counts.get(player_id, 0)

            if reward == 1:
                parallel_stats["agent_stats"][agent_name]["wins"] += 1

        # Update win rates
        for role in parallel_stats["role_stats"]:
            stats = parallel_stats["role_stats"][role]
            stats["win_rate"] = stats["wins"] / stats["games"] if stats["games"] > 0 else 0.0

        for agent_name in parallel_stats["agent_stats"]:
            stats = parallel_stats["agent_stats"][agent_name]
            stats["win_rate"] = stats["wins"] / stats["games"] if stats["games"] > 0 else 0.0

        # Update learning events
        for event in learning_events:
            parallel_stats["strategy_pool_stats"]["learning_events"].append({
                "timestamp": datetime.now().isoformat(),
                "game_id": parallel_stats["total_games"],
                "worker_id": game_result['worker_id'],
                **event
            })

def print_progress(current, total, start_time):
    """Print progress information"""
    elapsed = time.time() - start_time
    rate = current / elapsed if elapsed > 0 else 0
    eta = (total - current) / rate if rate > 0 else 0

    print(f"\rProgress: {current}/{total} ({current/total*100:.1f}%) | "
          f"Rate: {rate:.1f} games/sec | "
          f"ETA: {eta:.0f}s | "
          f"Elapsed: {elapsed:.0f}s", end="", flush=True)

def save_parallel_statistics(shared_pool):
    """Save parallel execution statistics"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create serializable copy
    serializable_stats = {
        "execution_summary": {
            "total_games": parallel_stats["total_games"],
            "completed_games": parallel_stats["completed_games"],
            "failed_games": parallel_stats["failed_games"],
            "success_rate": parallel_stats["completed_games"] / parallel_stats["total_games"] if parallel_stats["total_games"] > 0 else 0,
            "timestamp": timestamp
        },
        "game_results": {
            "mafia_wins": parallel_stats["mafia_wins"],
            "villager_wins": parallel_stats["villager_wins"],
            "role_stats": dict(parallel_stats["role_stats"])
        },
        "agent_performance": {},
        "strategy_pool_stats": parallel_stats["strategy_pool_stats"].copy(),
        "thread_performance": dict(parallel_stats["thread_stats"])
    }

    # Convert agent stats
    for agent_name, stats in parallel_stats["agent_stats"].items():
        serializable_stats["agent_performance"][agent_name] = {
            "games": stats["games"],
            "wins": stats["wins"],
            "win_rate": stats["win_rate"],
            "experiences_learned": stats["experiences_learned"],
            "roles": dict(stats["roles"])
        }

    # Save to file
    stats_file = f"parallel_enhanced_stats_{timestamp}.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_stats, f, indent=2, ensure_ascii=False)

    return stats_file

def main():
    """Main function for parallel enhanced self-play"""
    # Configuration
    NUM_GAMES = 16  # Total games to play
    NUM_WORKERS = 4  # Number of parallel threads

    print(f"Parallel Enhanced Self-Play Configuration:")
    print(f"  Total Games: {NUM_GAMES}")
    print(f"  Worker Threads: {NUM_WORKERS}")
    print(f"  Concurrent Games: {NUM_WORKERS}")
    print(f"  Strategy Pool: {STRATEGY_POOL_FILE}")
    print(f"  Logging: {ENABLE_LOGGING}")

    # Initialize shared strategy pool
    print("\nInitializing shared strategy pool...")
    shared_pool = setup_shared_strategy_pool()
    initial_experiences = shared_pool.get_statistics()["total_experiences"]
    print(f"Starting with {initial_experiences} experiences in pool")

    # Create task and result queues
    game_queue = queue.Queue()
    result_queue = queue.Queue()

    # Start worker threads
    print(f"\nStarting {NUM_WORKERS} worker threads...")
    workers = []
    for i in range(NUM_WORKERS):
        worker = GameWorker(i, game_queue, result_queue, shared_pool)
        worker.start()
        workers.append(worker)

    # Add games to queue
    print(f"Adding {NUM_GAMES} games to queue...")
    for game_id in range(1, NUM_GAMES + 1):
        game_queue.put((game_id, {"worker_id": None}))

    # Progress monitoring
    start_time = time.time()
    completed_count = 0

    print(f"\nStarting parallel execution...")
    print("=" * 80)

    # Process results
    while completed_count < NUM_GAMES:
        try:
            # Get result with timeout
            result = result_queue.get(timeout=10)

            # Update statistics
            update_statistics(result)
            completed_count += 1

            # Print progress
            print_progress(completed_count, NUM_GAMES, start_time)

        except queue.Empty:
            print(f"\nWarning: No results received in 10 seconds")
            continue

    print(f"\n\n{'='*80}")
    print("PARALLEL EXECUTION COMPLETED")
    print(f"{'='*80}")

    # Stop workers
    print("Stopping worker threads...")
    for _ in workers:
        game_queue.put(None)  # Poison pill

    for worker in workers:
        worker.join(timeout=5)

    # Final statistics
    total_time = time.time() - start_time
    final_pool_stats = shared_pool.get_statistics()

    print(f"\nExecution Summary:")
    print(f"  Total Time: {total_time:.1f} seconds")
    print(f"  Average Speed: {NUM_GAMES/total_time:.2f} games/second")
    print(f"  Efficiency: {(NUM_GAMES/total_time)/NUM_WORKERS*100:.1f}% per worker")

    print(f"\nFinal Strategy Pool:")
    print(f"  Total Experiences: {final_pool_stats['total_experiences']}")
    print(f"  New Experiences: {final_pool_stats['total_experiences'] - initial_experiences}")
    print(f"  Success Rate: {final_pool_stats['success_rate']*100:.1f}%")

    # Save statistics
    stats_file = save_parallel_statistics(shared_pool)
    print(f"\nStatistics saved to: {stats_file}")

    # Print detailed results
    print(f"\nGame Results:")
    print(f"  Total Games: {parallel_stats['total_games']}")
    print(f"  Completed: {parallel_stats['completed_games']}")
    print(f"  Failed: {parallel_stats['failed_games']}")
    print(f"  Mafia Wins: {parallel_stats['mafia_wins']} ({parallel_stats['mafia_wins']/parallel_stats['total_games']*100:.1f}%)")
    print(f"  Villager Wins: {parallel_stats['villager_wins']} ({parallel_stats['villager_wins']/parallel_stats['total_games']*100:.1f}%)")

    print(f"\nAgent Performance:")
    for agent_name, stats in parallel_stats["agent_stats"].items():
        print(f"  {agent_name}: {stats['wins']}/{stats['games']} ({stats['win_rate']*100:.1f}%), "
              f"Learned: {stats['experiences_learned']} experiences")

    print(f"\nThread Performance:")
    for thread_name, count in parallel_stats["thread_stats"].items():
        print(f"  {thread_name}: {count}")

if __name__ == "__main__":
    main()