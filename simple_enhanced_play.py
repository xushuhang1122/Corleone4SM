"""
Simple Enhanced Self-Play Script
A simplified version for quick testing of EnhancedMichael with strategy pool.
"""

import textarena as ta
from src.enhanced_agent import EnhancedMichael
from src.strategy_pool import StrategyExperiencePool
import time

def quick_enhanced_game():
    """Play a quick enhanced game to demonstrate learning"""
    print("=" * 50)
    print("QUICK ENHANCED SELF-PLAY DEMO")
    print("=" * 50)

    # Initialize shared strategy pool
    pool = StrategyExperiencePool("quick_strategy_pool.json")
    print(f"Strategy pool initialized with {pool.get_statistics()['total_experiences']} experiences")

    # Create enhanced agents
    agents = {
        0: EnhancedMichael("qwen3-8b", enable_learning=True),
        1: EnhancedMichael("qwen3-4b", enable_learning=True),
        2: EnhancedMichael("qwen3-8b", enable_learning=True),
        3: EnhancedMichael("qwen3-4b", enable_learning=True),
        4: EnhancedMichael("qwen3-8b", enable_learning=True),
        5: EnhancedMichael("qwen3-4b", enable_learning=True),
    }

    # Initialize environment
    env = ta.make(env_id="SecretMafia-v0")
    env.reset(num_players=len(agents))

    print("Starting enhanced game with learning agents...")
    start_time = time.time()

    # Game loop
    done = False
    turn_count = 0
    learning_events = 0

    while not done:
        turn_count += 1
        player_id, observation = env.get_observation()

        # Track learning before
        agent = agents[player_id]
        experiences_before = agent.get_learning_statistics()['total_experiences'] if hasattr(agent, 'get_learning_statistics') else 0

        # Agent takes action
        action = agent(observation)

        # Track learning after
        experiences_after = agent.get_learning_statistics()['total_experiences'] if hasattr(agent, 'get_learning_statistics') else 0
        if experiences_after > experiences_before:
            learning_events += 1
            print(f"  Turn {turn_count}: Player {player_id} learned new strategy!")

        done, step_info = env.step(action=action)

        if turn_count % 5 == 0:
            print(f"  Turn {turn_count}: Player {player_id} acted")

    rewards, game_info = env.close()
    end_time = time.time()

    # Results
    print(f"\nGame completed in {turn_count} turns ({end_time - start_time:.1f} seconds)")
    print(f"Learning events: {learning_events}")

    # Show final strategy pool stats
    final_pool_stats = pool.get_statistics()
    print(f"Final strategy pool: {final_pool_stats['total_experiences']} experiences")

    # Show individual agent learning
    print("\nAgent Learning Summary:")
    for player_id, agent in agents.items():
        if hasattr(agent, 'get_learning_statistics'):
            stats = agent.get_learning_statistics()
            print(f"  Player {player_id}: {stats['total_experiences']} experiences learned")

    # Determine winner
    mafia_wins = any(info["role"] == "Mafia" and reward == 1
                    for info, reward in zip(game_info.values(), rewards.values()))

    print(f"\nWinner: {'Mafia' if mafia_wins else 'Villagers'}")
    print(f"Final rewards: {rewards}")

    return learning_events > 0  # Return True if learning occurred

def main():
    """Run multiple quick enhanced games"""
    print("Starting Enhanced Self-Play Test Suite")
    print("This will test EnhancedMichael agents with strategy pool learning\n")

    # Run a few games
    games_with_learning = 0
    total_games = 3

    for game_num in range(1, total_games + 1):
        print(f"\n{'#'*60}")
        print(f"ENHANCED GAME {game_num}/{total_games}")
        print(f"{'#'*60}")

        try:
            learning_occurred = quick_enhanced_game()
            if learning_occurred:
                games_with_learning += 1
                print(f"✓ Game {game_num}: Learning occurred!")
            else:
                print(f"○ Game {game_num}: No new learning this game")
        except Exception as e:
            print(f"✗ Game {game_num} failed: {e}")

    # Summary
    print(f"\n{'='*60}")
    print("ENHANCED SELF-PLAY TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total games played: {total_games}")
    print(f"Games with learning: {games_with_learning}")
    print(f"Learning rate: {games_with_learning/total_games*100:.1f}%")

    if games_with_learning > 0:
        print(f"\n✓ Strategy pool learning is working!")
        print(f"Agents are successfully learning from gameplay and storing experiences.")
    else:
        print(f"\n○ No learning detected in this test run.")
        print(f"This could be normal - brilliant actions are rare by design.")

if __name__ == "__main__":
    main()