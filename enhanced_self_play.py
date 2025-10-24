"""
Enhanced Self-Play Script with Strategy Experience Pool
Uses EnhancedMichael agents for self-play games with learning capabilities.
All brilliant actions are automatically recorded to the strategy pool.
"""

import textarena as ta
from src.agent import LLMAgent
from src.game_logger import GameLogger, LoggedAgent
from src.enhanced_agent import EnhancedMichael, EnhancedLoggedAgent
from src.strategy_pool import StrategyExperiencePool
import sys
import io
import os
import json
import time
from collections import defaultdict, Counter
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Game configuration
MODEL_NAME = "EnhancedMichael-1.0"  # Name for enhanced agents
ENV_ID = "SecretMafia-v0"
ENABLE_LOGGING = True
LOG_DIR = "game_logs"
STRATEGY_POOL_FILE = "enhanced_strategy_pool.json"

# Agent configuration - all using EnhancedMichael with different models
AGENT_CONFIGS = {
    0: {"name": "Enhanced-Vito", "model": "deepseek-v3.1", "enable_learning": True},
    1: {"name": "Enhanced-Michael-1", "model": "deepseek-v3.1", "enable_learning": True},
    2: {"name": "Enhanced-Michael-2", "model": "deepseek-v3.1", "enable_learning": True},
    3: {"name": "Enhanced-Michael-3", "model": "deepseek-v3.1", "enable_learning": True},
    4: {"name": "Enhanced-Michael-4", "model": "deepseek-v3.1", "enable_learning": True},
    5: {"name": "Enhanced-Michael-5", "model": "deepseek-v3.1", "enable_learning": True},
}

# Enhanced game statistics tracking
enhanced_stats = {
    "total_games": 0,
    "mafia_wins": 0,
    "villager_wins": 0,
    "role_stats": defaultdict(lambda: {"wins": 0, "games": 0, "win_rate": 0.0}),
    "reasons": Counter(),
    "invalid_moves": 0,
    "agent_stats": defaultdict(lambda: {
        "wins": 0, "games": 0, "win_rate": 0.0,
        "roles": Counter(), "experiences_learned": 0
    }),
    "strategy_pool_stats": {
        "total_experiences": 0,
        "successful_experiences": 0,
        "experiences_by_role": defaultdict(int),
        "learning_events": []
    }
}

def initialize_enhanced_agents():
    """Initialize enhanced agents for a new game"""
    agents = {}
    for player_id, config in AGENT_CONFIGS.items():
        print(f"Initializing {config['name']} with model {config['model']}")
        agents[player_id] = EnhancedMichael(
            model_name=config['model'],
            enable_learning=config['enable_learning']
        )
    return agents

def get_agent_config(player_id):
    """Get agent configuration based on player ID"""
    return AGENT_CONFIGS.get(player_id, {"name": f"agent_{player_id}", "model": "unknown"})

def setup_shared_strategy_pool():
    """Setup shared strategy pool for all agents"""
    pool = StrategyExperiencePool(STRATEGY_POOL_FILE)

    # Add some initial brilliant experiences if pool is empty
    if pool.get_statistics()["total_experiences"] == 0:
        print("Adding initial brilliant experiences to strategy pool...")
        add_initial_experiences(pool)

    return pool

def add_initial_experiences(pool):
    """Add initial brilliant experiences from configuration file"""
    try:
        with open("initial_experiences.json", 'r', encoding='utf-8') as f:
            config = json.load(f)

        initial_experiences = config.get("experiences", [])

        for exp in initial_experiences:
            pool.add_experience(**exp)

        print(f"Added {len(initial_experiences)} initial experiences from configuration file")

    except FileNotFoundError:
        print("Warning: initial_experiences.json not found, using hardcoded experiences")
        # Fallback to hardcoded experiences
        fallback_experiences = [
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
            }
        ]

        for exp in fallback_experiences:
            pool.add_experience(**exp)

        print(f"Added {len(fallback_experiences)} fallback initial experiences")

    except Exception as e:
        print(f"Error loading initial experiences: {e}")
        print("Proceeding without initial experiences")

def play_enhanced_single_game(game_id, logger, shared_pool):
    """Play a single enhanced game and return results"""
    print(f"\n{'='*60}")
    print(f"Starting Enhanced Game {game_id}")
    print(f"{'='*60}")

    # Initialize enhanced agents for this game
    raw_agents = initialize_enhanced_agents()

    # Store initial experience counts for learning tracking
    initial_experience_counts = {}
    for player_id, agent in raw_agents.items():
        if hasattr(agent, 'get_learning_statistics'):
            initial_experience_counts[player_id] = agent.get_learning_statistics().get('total_experiences', 0)

    # Wrap agents with enhanced logging if enabled
    if ENABLE_LOGGING:
        agents = {}
        for player_id, agent in raw_agents.items():
            agents[player_id] = EnhancedLoggedAgent(agent, logger, player_id, enable_learning=True)
    else:
        agents = raw_agents

    # Initialize the environment
    env = ta.make(env_id=ENV_ID)
    env.reset(num_players=len(agents))

    # Start logging session
    if ENABLE_LOGGING:
        session_id = logger.start_session(
            env_id=ENV_ID,
            agents=raw_agents,
            num_players=len(agents),
            additional_info={
                "script": "enhanced_self_play.py",
                "game_id": game_id,
                "model_name": MODEL_NAME,
                "description": "Enhanced self-play with strategy pool learning",
                "strategy_pool_experiences": shared_pool.get_statistics()["total_experiences"],
                "agent_configs": {pid: config for pid, config in AGENT_CONFIGS.items() if pid < len(agents)}
            }
        )
    else:
        session_id = None

    # Main game loop
    done = False
    turn_count = 0
    learning_events = []

    while not done:
        turn_count += 1
        player_id, observation = env.get_observation()

        # Track learning before action
        if hasattr(agents[player_id], 'base_agent') and hasattr(agents[player_id].base_agent, 'get_learning_statistics'):
            stats_before = agents[player_id].base_agent.get_learning_statistics()

        action = agents[player_id](observation)

        # Track learning after action
        if hasattr(agents[player_id], 'base_agent') and hasattr(agents[player_id].base_agent, 'get_learning_statistics'):
            stats_after = agents[player_id].base_agent.get_learning_statistics()
            if stats_after.get('total_experiences', 0) > stats_before.get('total_experiences', 0):
                learning_events.append({
                    'turn': turn_count,
                    'player_id': player_id,
                    'agent_name': get_agent_config(player_id)['name'],
                    'new_experiences': stats_after.get('total_experiences', 0) - stats_before.get('total_experiences', 0)
                })
                print(f"  Learning Event: Player {player_id} ({get_agent_config(player_id)['name']}) learned new strategy!")

        done, step_info = env.step(action=action)

        if not ENABLE_LOGGING and turn_count % 10 == 0:
            print(f"  Turn {turn_count}: Player {player_id} ({get_agent_config(player_id)['name']}) acted")

    rewards, game_info = env.close()

    # Log results if enabled
    if ENABLE_LOGGING:
        logger.log_results(rewards, game_info)
        logger.end_session()
        print(f"Game {game_id} completed and logged: {session_id}")

    print(f"  Total turns: {turn_count}")
    print(f"  Learning events in this game: {len(learning_events)}")

    # Track experience learning for each agent
    final_experience_counts = {}
    for player_id, agent in raw_agents.items():
        if hasattr(agent, 'get_learning_statistics'):
            final_stats = agent.get_learning_statistics()
            final_count = final_stats.get('total_experiences', 0)
            initial_count = initial_experience_counts.get(player_id, 0)
            experiences_learned = final_count - initial_count
            final_experience_counts[player_id] = experiences_learned

            if experiences_learned > 0:
                print(f"  {get_agent_config(player_id)['name']} learned {experiences_learned} new experiences")

    return rewards, game_info, raw_agents, learning_events, final_experience_counts

def update_enhanced_statistics(rewards, game_info, agents, learning_events, experience_counts, shared_pool):
    """Update enhanced game statistics based on game results"""
    enhanced_stats["total_games"] += 1

    # Count wins by team
    mafia_win = any(info["role"] in ["Mafia"] and reward == 1
                   for info, reward in zip(game_info.values(), rewards.values()))

    if mafia_win:
        enhanced_stats["mafia_wins"] += 1
    else:
        enhanced_stats["villager_wins"] += 1

    # Track reasons for game end
    for player_info in game_info.values():
        enhanced_stats["reasons"][player_info["reason"]] += 1
        if player_info["invalid_move"]:
            enhanced_stats["invalid_moves"] += 1

    # Track role-based statistics
    for player_id, info in game_info.items():
        role = info["role"]
        reward = rewards[player_id]

        enhanced_stats["role_stats"][role]["games"] += 1
        if reward == 1:  # Win
            enhanced_stats["role_stats"][role]["wins"] += 1

        # Update win rate
        games = enhanced_stats["role_stats"][role]["games"]
        wins = enhanced_stats["role_stats"][role]["wins"]
        enhanced_stats["role_stats"][role]["win_rate"] = wins / games if games > 0 else 0.0

        # Track enhanced agent-based statistics
        agent_config = get_agent_config(player_id)
        agent_name = agent_config["name"]
        enhanced_stats["agent_stats"][agent_name]["games"] += 1
        enhanced_stats["agent_stats"][agent_name]["roles"][role] += 1
        enhanced_stats["agent_stats"][agent_name]["experiences_learned"] += experience_counts.get(player_id, 0)

        if reward == 1:  # Win
            enhanced_stats["agent_stats"][agent_name]["wins"] += 1

        # Update agent win rate
        agent_games = enhanced_stats["agent_stats"][agent_name]["games"]
        agent_wins = enhanced_stats["agent_stats"][agent_name]["wins"]
        enhanced_stats["agent_stats"][agent_name]["win_rate"] = agent_wins / agent_games if agent_games > 0 else 0.0

    # Update strategy pool statistics
    pool_stats = shared_pool.get_statistics()
    enhanced_stats["strategy_pool_stats"]["total_experiences"] = pool_stats["total_experiences"]
    enhanced_stats["strategy_pool_stats"]["successful_experiences"] = int(pool_stats["success_rate"] * pool_stats["total_experiences"])
    enhanced_stats["strategy_pool_stats"]["experiences_by_role"] = pool_stats["experiences_by_role"].copy()

    # Record learning events
    for event in learning_events:
        enhanced_stats["strategy_pool_stats"]["learning_events"].append({
            "timestamp": datetime.now().isoformat(),
            "game_id": enhanced_stats["total_games"],
            **event
        })

def print_enhanced_statistics(shared_pool):
    """Print comprehensive enhanced game statistics"""
    print(f"\n{'='*80}")
    print(f"ENHANCED SELF-PLAY STATISTICS AFTER {enhanced_stats['total_games']} GAMES")
    print(f"{'='*80}")

    # Overall win rates
    print(f"\nOverall Results:")
    print(f"  Total Games: {enhanced_stats['total_games']}")
    print(f"  Mafia Wins: {enhanced_stats['mafia_wins']} "
          f"({enhanced_stats['mafia_wins']/enhanced_stats['total_games']*100:.1f}%)")
    print(f"  Villager Wins: {enhanced_stats['villager_wins']} "
          f"({enhanced_stats['villager_wins']/enhanced_stats['total_games']*100:.1f}%)")

    # Enhanced agent-based statistics
    print(f"\nEnhanced Agent Statistics:")
    for agent_name, stats in enhanced_stats["agent_stats"].items():
        print(f"  {agent_name}:")
        print(f"    Games: {stats['games']}, Wins: {stats['wins']} "
              f"({stats['win_rate']*100:.1f}%)")
        print(f"    Experiences Learned: {stats['experiences_learned']}")
        print(f"    Roles Played: {dict(stats['roles'])}")

    # Role-based statistics
    print(f"\nRole-based Statistics:")
    for role, stats in enhanced_stats["role_stats"].items():
        print(f"  {role}: {stats['wins']}/{stats['games']} "
              f"({stats['win_rate']*100:.1f}%)")

    # Strategy pool statistics
    pool_stats = enhanced_stats["strategy_pool_stats"]
    print(f"\nStrategy Pool Statistics:")
    print(f"  Total Experiences: {pool_stats['total_experiences']}")
    print(f"  Successful Experiences: {pool_stats['successful_experiences']}")
    if pool_stats['total_experiences'] > 0:
        success_rate = pool_stats['successful_experiences'] / pool_stats['total_experiences']
        print(f"  Success Rate: {success_rate*100:.1f}%")
    print(f"  Experiences by Role: {dict(pool_stats['experiences_by_role'])}")

    # Recent learning events
    recent_learning = pool_stats["learning_events"][-10:]  # Last 10 events
    if recent_learning:
        print(f"\nRecent Learning Events (Last 10):")
        for event in recent_learning:
            print(f"  Game {event['game_id']}, Turn {event['turn']}: "
                  f"{event['agent_name']} learned {event['new_experiences']} new strategies")

    # Game end reasons
    print(f"\nGame End Reasons:")
    for reason, count in enhanced_stats["reasons"].most_common():
        print(f"  {reason}: {count}")

    # Invalid moves
    print(f"\nInvalid Moves: {enhanced_stats['invalid_moves']}")

def save_enhanced_statistics(shared_pool):
    """Save enhanced statistics to file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save game statistics
    stats_file = f"enhanced_stats_{timestamp}.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        # Convert defaultdict to regular dict for JSON serialization
        serializable_stats = {
            "total_games": enhanced_stats["total_games"],
            "mafia_wins": enhanced_stats["mafia_wins"],
            "villager_wins": enhanced_stats["villager_wins"],
            "role_stats": dict(enhanced_stats["role_stats"]),
            "reasons": dict(enhanced_stats["reasons"]),
            "invalid_moves": enhanced_stats["invalid_moves"],
            "agent_stats": dict(enhanced_stats["agent_stats"]),
            "strategy_pool_stats": enhanced_stats["strategy_pool_stats"].copy(),
            "timestamp": timestamp
        }

        # Convert Counter objects to regular dicts
        for agent_name, stats in serializable_stats["agent_stats"].items():
            stats["roles"] = dict(stats["roles"])

        serializable_stats["reasons"] = dict(enhanced_stats["reasons"])

        json.dump(serializable_stats, f, indent=2, ensure_ascii=False)

    print(f"\nEnhanced statistics saved to: {stats_file}")
    return stats_file

def main():
    """Main function to run enhanced self-play games"""
    NUM_GAMES = 1  # Number of enhanced self-play games

    print(f"Starting {NUM_GAMES} enhanced self-play games of Secret Mafia...")
    print(f"All agents will use EnhancedMichael with strategy pool learning")
    if ENABLE_LOGGING:
        print(f"Enhanced logging enabled. Logs will be saved to: {LOG_DIR}")
    print(f"Strategy pool will be saved to: {STRATEGY_POOL_FILE}")

    # Initialize shared strategy pool
    print("\nInitializing shared strategy pool...")
    shared_pool = setup_shared_strategy_pool()

    # Initialize enhanced game logger
    logger = GameLogger(log_dir=LOG_DIR, enabled=ENABLE_LOGGING)

    # Play multiple enhanced games
    start_time = time.time()
    for game_id in range(1, NUM_GAMES + 1):
        try:
            print(f"\n{'#'*80}")
            print(f"ENHANCED GAME {game_id}/{NUM_GAMES}")
            print(f"{'#'*80}")

            rewards, game_info, agents, learning_events, experience_counts = play_enhanced_single_game(
                game_id, logger, shared_pool
            )
            update_enhanced_statistics(rewards, game_info, agents, learning_events, experience_counts, shared_pool)

            # Show intermediate strategy pool stats
            pool_stats = shared_pool.get_statistics()
            print(f"  Strategy pool now has {pool_stats['total_experiences']} experiences")

        except Exception as e:
            print(f"Error in enhanced game {game_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    end_time = time.time()
    total_time = end_time - start_time

    # Print final enhanced statistics
    print_enhanced_statistics(shared_pool)

    # Save statistics
    stats_file = save_enhanced_statistics(shared_pool)

    # Display logging statistics if enabled
    if ENABLE_LOGGING:
        stats = logger.get_session_stats()
        print(f"\n{'='*60}")
        print("ENHANCED LOGGING STATISTICS")
        print(f"{'='*60}")
        print(f"Total sessions logged: {stats['total_sessions']}")
        print(f"Sessions by environment: {stats['sessions_by_env']}")
        print(f"Log directory: {LOG_DIR}")
        print(f"Latest session: {stats.get('latest_session', {}).get('session_id', 'None')}")

    # Performance summary
    print(f"\n{'='*60}")
    print("ENHANCED SELF-PLAY SUMMARY")
    print(f"{'='*60}")
    print(f"Total Games Played: {NUM_GAMES}")
    print(f"Total Time: {total_time:.1f} seconds ({total_time/NUM_GAMES:.1f} seconds per game)")
    print(f"Average Learning Events per Game: {len(enhanced_stats['strategy_pool_stats']['learning_events'])/NUM_GAMES:.1f}")
    print(f"Final Strategy Pool Size: {shared_pool.get_statistics()['total_experiences']} experiences")
    print(f"Statistics File: {stats_file}")

if __name__ == "__main__":
    main()