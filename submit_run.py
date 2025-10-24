"""
Submission Entry Point for Social Detection Track
This script connects the enhanced Michael agent to the online competition.
Environment: SecretMafia-v0
"""

import textarena as ta
from src.huggingface_agent import HuggingFaceMichael
import sys
import io
import os

# Set encoding for stdout/stderr
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Model configuration
MODEL_NAME = "Michael-v1.0"  # Replace with your model name
# The name is used to identify your agent in the online arena and leaderboard.
# It should be unique and descriptive.
# For different versions of your agent, you should use different names.

MODEL_DESCRIPTION = "Enhanced Michael agent with strategy experience pool"

# Team configuration - replace with your actual team hash
TEAM_HASH = "MG25-F5C82328D3"  # Replace with your team hash

# Model configuration
# You can change this to other models like "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B", etc.
# Make sure the model is available on HuggingFace Hub
HUGGINGFACE_MODEL_NAME = "Qwen/Qwen3-8B"

def main():
    """
    Main function to run the enhanced Michael agent in the competition environment.
    """
    print(f"Initializing Enhanced Michael agent with model: {HUGGINGFACE_MODEL_NAME}")
    print(f"Agent Name: {MODEL_NAME}")
    print(f"Team Hash: {TEAM_HASH}")

    try:
        # Initialize the enhanced Michael agent with HuggingFace model
        agent = HuggingFaceMichael(
            model_name=HUGGINGFACE_MODEL_NAME,
            enable_learning=True  # Enable strategy experience pool learning
        )

        print("Agent initialized successfully!")

        # Create the competition environment
        env = ta.make_mgc_online(
            track="Social Detection",
            model_name=MODEL_NAME,
            model_description=MODEL_DESCRIPTION,
            team_hash=TEAM_HASH,
            agent=agent,
            small_category=True  # Set to True to participate in the efficient division
        )

        print("Competition environment created!")

        # Reset environment - always set to 1 when playing online
        env.reset(num_players=1)
        print("Environment reset completed!")

        # Main game loop
        done = False
        turn_count = 0

        print("Starting game loop...")

        while not done:
            turn_count += 1
            print(f"\n--- Turn {turn_count} ---")

            # Get observation
            player_id, observation = env.get_observation()
            print(f"Received observation for player {player_id}")

            # Get agent action
            print("Getting agent action...")
            action = agent(observation)
            print(f"Agent action: {action}")

            # Submit action
            print("Submitting action to environment...")
            done, step_info = env.step(action=action)
            print(f"Step completed. Done: {done}")
            print(f"Step info: {step_info}")

        # Close environment and get results
        print("\nGame completed!")
        rewards, game_info = env.close()

        print(f"Final rewards: {rewards}")
        print(f"Game info: {game_info}")

        # Print learning statistics if available
        if hasattr(agent, 'get_learning_statistics'):
            stats = agent.get_learning_statistics()
            print(f"Learning statistics: {stats}")

        print("Submission completed successfully!")

    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)