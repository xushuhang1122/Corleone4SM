"""
Track 1: Social Detection Track
This script connects your agent to the online competition for the social detection track.
Environment: SecretMafia-v0
"""

import textarena as ta
from agent import LLMAgent
from xushuhang.agents.family import Vito

MODEL_NAME = "Test LLM agent - Track 1" # Replace with your model name
# The name is used to identify your agent in the online arena and leaderboard.
# It should be unique and descriptive.
# For different versions of your agent, you should use different names.
MODEL_DESCRIPTION = "This agent is for Track 1 - Social Detection (SecretMafia-v0)."
team_hash = "MG25-XXXXXXXXXX" # Replace with your team hash

# Initialize your agent
# agent = LLMAgent(model_name="Qwen/Qwen3-4B")
agent = Vito()

env = ta.make_mgc_online(
    track="Social Detection", 
    model_name=MODEL_NAME,
    model_description=MODEL_DESCRIPTION,
    team_hash=team_hash,
    agent=agent,
    small_category=False  # Set to True to participate in the efficient division
)
env.reset(num_players=1) # always set to 1 when playing online, even when playing multiplayer games.

done = False
while not done:
    player_id, observation = env.get_observation()
    action = agent(observation)
    done, step_info = env.step(action=action)

rewards, game_info = env.close() 