"""
Enhanced Agent with HuggingFace Model Integration
Integrates brilliant action learning with local HuggingFace models.
"""

import json
import re
import time
import torch
from typing import Dict, List, Any, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from .strategy_pool import StrategyExperiencePool
from .game_state_analyzer import GameStateAnalyzer
from .brilliant_action_detector import BrilliantActionDetector
from .strategy_matcher import StrategyMatcher


class HuggingFaceMichael:
    """
    Enhanced Michael agent with HuggingFace model integration and strategy experience pool.
    Learns from brilliant actions and uses past experiences to improve strategy.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B", enable_learning: bool = True):
        """
        Initialize Enhanced Michael with HuggingFace model and experience pool capabilities.

        Args:
            model_name: Name of the HuggingFace model to use
            enable_learning: Whether to enable learning from experiences
        """
        self.model_name = model_name
        self.is_initialized = False
        self.init_info = None
        self.belief = ""
        self.strategy = ""
        self.observation_history = []
        self.round = 0
        self.enable_learning = enable_learning

        # Initialize HuggingFace model
        self._init_huggingface_model()

        if enable_learning:
            # Initialize experience pool components
            self.experience_pool = StrategyExperiencePool()
            self.state_analyzer = GameStateAnalyzer(self.api, self.model_name)
            self.brilliant_action_detector = BrilliantActionDetector(
                self.experience_pool, self.state_analyzer, self.api, self.model_name
            )
            self.strategy_matcher = StrategyMatcher(
                self.experience_pool, self.state_analyzer, self.api, self.model_name
            )

            # Track previous observation for outcome evaluation
            self.previous_observation = None
            self.pending_action_keys = []

            print(f"Enhanced Michael initialized with {model_name} and strategy experience pool")
        else:
            print(f"Enhanced Michael initialized with {model_name} without learning (learning disabled)")

    def _init_huggingface_model(self):
        """Initialize HuggingFace model and tokenizer."""
        try:
            print(f"Loading HuggingFace model: {self.model_name}")

            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            # Add pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )

            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=torch.float16,
                device_map="auto"
            )

            print(f"HuggingFace model {self.model_name} loaded successfully")

        except Exception as e:
            print(f"Error loading HuggingFace model: {e}")
            raise

    def api(
        self,
        input_messages: Optional[List[Dict]] = None,
        temperature: float = 0.4,
        max_tokens: int = 1024,
    ):
        """
        API method using HuggingFace model for text generation.

        Args:
            input_messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate

        Returns:
            Generated text response
        """
        if input_messages is None:
            raise ValueError("messages should not be None!")

        try:
            # Format messages for chat template
            if hasattr(self.tokenizer, 'apply_chat_template'):
                # Use chat template if available
                prompt = self.tokenizer.apply_chat_template(
                    input_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # Fallback to simple concatenation
                prompt = ""
                for message in input_messages:
                    if message["role"] == "system":
                        prompt += f"System: {message['content']}\n\n"
                    elif message["role"] == "user":
                        prompt += f"User: {message['content']}\n\n"

                prompt += "Assistant: "

            print(f"Generating response with {self.model_name}...")

            # Generate response
            outputs = self.generator(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )

            response = outputs[0]['generated_text'].strip()
            print(f"Generated response length: {len(response)} characters")

            return response

        except Exception as e:
            print(f"Error in HuggingFace API call: {e}")
            # Fallback response
            return "I apologize, but I encountered an error while processing your request."

    def __call__(self, observation: str) -> str:
        """
        Enhanced call method with experience pool integration.

        Args:
            observation: Current game observation

        Returns:
            Agent action/response
        """
        print("\n=================================== ENHANCED MICHAEL (HF) ====================================")
        print("Observation:", observation)

        # Parse observation
        obs_list = json.loads(observation) if isinstance(observation, str) and observation.startswith('[') else observation

        # Complete any pending action evaluations using new observation
        if self.enable_learning and self.previous_observation:
            self._complete_pending_evaluations(observation)

        # First observation - initialization
        if not self.is_initialized:
            self.init_info = self.parse_initialization_info(observation)
            self.init_identity = self.generate_identity_prompt(self.init_info)
            self.belief = self.generate_belief_prompt(self.init_info)
            self.strategy = "No strategy set yet. Will develop based on game progress."
            self.is_initialized = True
            print("Enhanced Michael: Game information initialized.\n")

        # Determine current round and phase
        current_round = self.round % 5
        if current_round == 0:
            if self.init_info["role"] == "Villager":
                self.round += 1
                phase = "day_speak"
            else:
                phase = "night"
        elif current_round in [1, 2, 3]:
            phase = "day_speak"
        else:
            phase = "day_vote"
        self.round += 1

        print(f"Current Round: {current_round}, Phase: {phase}")

        # Format observation
        formatted_obs = self.parse_observation_events(obs_list) if isinstance(obs_list, list) else observation
        self.observation_history.append(formatted_obs)

        # Get relevant experiences for strategy enhancement
        strategy_guidance = ""
        if self.enable_learning:
            strategy_guidance = self._get_strategy_guidance(formatted_obs, phase)

        # Step 1: Analyze new information (with experience guidance)
        analysis = self._enhanced_analyze(formatted_obs, strategy_guidance)

        # Step 2: Update beliefs
        self.belief = self.parse_llm_response(
            self.api(input_messages=[
                {"role": "system", "content": self.prompt_system()},
                {"role": "user", "content": self.prompt_belief(analysis, self.belief)}
            ]),
            "#BELIEF:"
        )

        # Step 3: Update strategy (with experience guidance)
        self.strategy = self._enhanced_strategy_update(analysis, self.belief, strategy_guidance)

        # Step 4: Generate final action/speech
        if phase == "day_speak":
            final_response = self.parse_llm_response(
                self.api(input_messages=[
                    {"role": "system", "content": self.prompt_system()},
                    {"role": "user", "content": self.prompt_talk(self.belief, self.strategy)}
                ]),
                "#FINAL:")
            final_output = final_response
        else:
            final_response = self.parse_llm_response(
                self.api(input_messages=[
                    {"role": "system", "content": self.prompt_system()},
                    {"role": "user", "content": self.prompt_vote(self.belief, self.strategy)}
                ]),
                "#FINAL:")

            bracket_match = re.search(r'\[(\d+)\]', final_response)
            if bracket_match:
                final_output = f"[{bracket_match.group(1)}]"
            else:
                patterns = [
                    r'vote[^\d]{0,10}(\d+)',
                    r'detect[^\d]{0,10}(\d+)',
                    r'eliminate[^\d]{0,10}(\d+)',
                    r'kill[^\d]{0,10}(\d+)',
                    r'protect[^\d]{0,10}(\d+)',
                    r'rescue[^\d]{0,10}(\d+)',
                    r'investigate[^\d]{0,10}(\d+)'
                ]

                reversed_text = final_response[::-1]
                found_number = None
                for pattern in patterns:
                    reversed_pattern = pattern[::-1]
                    match = re.search(reversed_pattern, reversed_text)
                    if match:
                        found_number = match.group(1)[::-1]
                        break

                if found_number:
                    final_output = f"[{found_number}]"

        # Evaluate this action for potential brilliance (if learning enabled)
        if self.enable_learning:
            self._evaluate_action_for_brilliance(
                formatted_obs, final_output, phase, current_round
            )

        # Store current observation for next evaluation cycle
        if self.enable_learning:
            self.previous_observation = observation

        print("response:\n", final_response)
        print("\nFINAL OUTPUT:\n", final_output)
        print("\n\n" + "=" * 60 + "\n\n\n\n\n\n")

        return final_output

    # Include all the same methods as the original EnhancedMichael class
    def parse_observation_events(self, observation: List[List]) -> str:
        """
        Parse observation list into formatted event statements

        Args:
            observation: List of events, each event is [speaker_id, content, message_type]
                        speaker_id: -1 for system, 0-N for players
                        content: message content
                        message_type: typically 2 for speech, 4 for system announcement

        Returns:
            Formatted event description text
        """
        prompt_parts = []

        for event in observation:
            if len(event) < 2:
                continue

            speaker_id = event[0]
            content = event[1]

            # Check if content contains a vote pattern like [0], [1], etc.
            vote_match = re.search(r'^\[(\d+)\]$', content.strip())

            if speaker_id == -1:
                # System message
                prompt_parts.append(f"SYSTEM: {content}")
            elif vote_match:
                # Vote action
                voted_player = vote_match.group(1)
                prompt_parts.append(f"Player {speaker_id} VOTED: [Player {voted_player}]")
            else:
                # Regular speech
                # Check if it's wrapped in brackets (internal thoughts)
                if content.strip().startswith('[') and content.strip().endswith(']'):
                    prompt_parts.append(f"Player {speaker_id} (internal): {content}")
                else:
                    prompt_parts.append(f"Player {speaker_id}: {content}")

        return "\n".join(prompt_parts)

    def parse_initialization_info(self, observation_text: str) -> Dict:
        """Parse initialization information from observation text."""
        init_info = {
            "player_id": None,
            "role": None,
            "team": None,
            "description": None,
            "all_players": [],
            "teammates": []
        }

        # Extract player ID
        player_match = re.search(r'You are Player (\d+)', observation_text)
        if player_match:
            init_info["player_id"] = int(player_match.group(1))

        # Extract role information
        role_match = re.search(r'Your role: (.+)', observation_text)
        if role_match:
            init_info["role"] = role_match.group(1).strip()

        # Extract team information
        team_match = re.search(r'Team: (.+)', observation_text)
        if team_match:
            init_info["team"] = team_match.group(1).strip()

        # Extract description information
        desc_match = re.search(r'Description: (.+?)(?:\n\n|Players:)', observation_text, re.DOTALL)
        if desc_match:
            init_info["description"] = desc_match.group(1).strip()

        # Extract all players list
        players_match = re.search(r'Players: (.+)', observation_text)
        if players_match:
            players_str = players_match.group(1)
            # Extract all Player X format players
            player_ids = re.findall(r'Player (\d+)', players_str)
            init_info["all_players"] = [int(pid) for pid in player_ids]

        # Extract teammate information
        teammates_match = re.search(r'Your teammates are: (.+?)\.', observation_text)
        if teammates_match:
            teammates_str = teammates_match.group(1)
            teammate_ids = re.findall(r'Player (\d+)', teammates_str)
            init_info["teammates"] = [int(tid) for tid in teammate_ids]

        return init_info

    def generate_identity_prompt(self, init_info: Dict) -> str:
        """Generate identity description prompt from initialization info."""
        prompt_parts = []

        prompt_parts.append("=== YOUR IDENTITY ===")
        prompt_parts.append(f"Player ID: Player {init_info['player_id']}")
        prompt_parts.append(f"Role: {init_info['role']}")
        prompt_parts.append(f"Team: {init_info['team']}")
        prompt_parts.append(f"Description: {init_info['description']}")

        if init_info['role'] == "Mafia":
            prompt_parts.append("Goals: Try to deceive other players to conceal yourself and your companions until enough villagers are eliminated.")
        elif init_info['role'] == "A regular villager":
            prompt_parts.append("Goals: Try to identify and eliminate all mafia members through discussion and voting.")
        elif init_info['role'] == "Detective":
            prompt_parts.append("Goals: Try to Protect yourself and identify mafia members through investigation and help villagers eliminate them.")
        elif init_info['role'] == "Doctor":
            prompt_parts.append("Goals: Try to protect key villagers, especially yourself and the detective, from being eliminated by the mafia during the night.")

        return "\n".join(prompt_parts)

    def generate_belief_prompt(self, init_info: Dict, custom_beliefs: Optional[Dict[int, Dict]] = None) -> str:
        """Generate belief information prompt for all players."""
        prompt_parts = []
        prompt_parts.append("=== PLAYER BELIEFS ===")

        # If no custom_beliefs provided, initialize based on init_info
        if custom_beliefs is None:
            custom_beliefs = {}

            for player_id in init_info['all_players']:
                if player_id == init_info['player_id']:
                    # Self
                    custom_beliefs[player_id] = {
                        "role": init_info['role'],
                        "status": "confirmed",
                        "notes": "This is yourself.",
                        "alive": True
                    }
                elif player_id in init_info['teammates']:
                    # Teammates
                    custom_beliefs[player_id] = {
                        "role": init_info['role'],  # same role as teammates
                        "status": "confirmed",
                        "notes": "Confirmed by system.",
                        "alive": True
                    }
                else:
                    # Other players
                    custom_beliefs[player_id] = {
                        "role": "Unknown",
                        "status": "unknown",
                        "notes": "There is currently no further information available.",
                        "alive": True
                    }

        # Sort by player ID
        sorted_players = sorted(custom_beliefs.keys())

        for player_id in sorted_players:
            belief = custom_beliefs[player_id]
            status = belief["status"]

            # Format status and role
            if status == "confirmed":
                role_info = f"Role: CONFIRMED as {belief['role']}"
            elif status == "suspected":
                role_info = f"Role: SUSPECTED as {belief['role']}"
            else:
                role_info = "Role: UNKNOWN"

            # Add alive/dead status
            life_status = "ALIVE" if belief.get("alive", True) else "DEAD"

            # Build entry with consistent formatting
            entry = f"Player {player_id}: {role_info} | Status: {life_status}"

            if belief.get("notes"):
                entry += f" | Notes: {belief['notes']}"

            prompt_parts.append(entry)

        return "\n".join(prompt_parts)

    def prompt_system(self) -> str:
        """Generate system prompt."""
        ret = f"""
        You are participating in the game Secret Mafia and playing one of the roles.\n
        The game will start at night and alternate between night and day. At night, Mafia will secretly eliminate players, detectives can investigate a player's identity, and doctors can choose to protect a player from being eliminated by Mafia. During each daytime stage, players will have 3 rounds of discussion and then vote to eliminate the player with the most votes.
        \n\n
        Here are some information about this game:\n
        {self.init_identity}\n

    """
        return ret

    def prompt_analyze(self, observation) -> str:
        """Generate analysis prompt."""
        ret = f"""
    Your actions in each round are divided into four steps: 1 Analyze newly acquired information; 2. Update the identification of other players' identities; 3. Update your own strategy; 4. Decide on your own speech or action.
    Now it is step 1. Analyze newly acquired information.

    # You got these new information:
    {observation}

    # Please follow the steps:
    1. What information does the system declaration reflect? You may need to analyze this - for example, if a player is eliminated at night, it can reflect Mafia's strategy, and if no player is eliminated at night, it indicates that the doctor successfully protected a player.
    2. For other players' comments, try to empathize with their perspective one by one: why do they speak like this? What is the purpose? This may reflect their identity or strategy.
    3. Summarize your analysis results and start with a symbol: "#SUMMARY:"
    """
        return ret

    def prompt_belief(self, analysis, belief) -> str:
        """Generate belief update prompt."""
        ret = f"""
    Your actions in each round are divided into four steps: 1 Analyze newly acquired information; 2. Update the identification of other players' identities; 3. Update your own strategy; 4. Decide on your own speech or action.
    Now it is step 2. You should update your beliefs.

    Here is your analysis results:
    {analysis}


    # Previous belief:
    {belief}

    # Please follow the steps:
    1. Based on system message, which players' survival status needs to be modified?
    2. Based on your analysis just now, which players' identities can be guessed? Note that identity confirmation can only be set through system messages from Mafia and Detection, otherwise you can only suspect their roles.
    3. Modify your BELIEF and generate a new BELIEF, maintain the format: [player_id: player identity guess | survival status | explanation of identity guess and elimination reason.], starting with the symbol: "#BELIEF:"

    """
        return ret

    def prompt_strategy(self, analysis, belief, strategy) -> str:
        """Generate strategy update prompt."""
        ret = f"""
    Your actions in each round are divided into four steps: 1 Analyze newly acquired information; 2. Update the identification of other players' identities; 3. Update your own strategy; 4. Decide on your own speech or action.
    Now it is step 3. Please refer to your goals, analysis, and beliefs, then decide your strategy.

    # Your analysis:
    {analysis}

    # Your belief:
    {belief}

    # Your previous strategy:
    {strategy}

    # Please follow the steps:
    1. What is your goal?
    2. Based on your analysis and belief, what is your strategy? For example, you can decide whether to claim which character you are, encourage everyone to expel which player, explain your words and actions to everyone, and so on.
    3. Generate a new STRATEGY, starting with the symbol: "#STRATEGY:"

    """
        return ret

    def prompt_talk(self, belief, strategy) -> str:
        """Generate speech prompt."""
        ret = f"""
    Your actions in each round are divided into four steps: 1 Analyze newly acquired information; 2. Update the identification of other players' identities; 3. Update your own strategy; 4. Decide on your own speech or action.
    Now it is step 4. Decide on your own speech or action.

    You need to speak now, you need to generate the final speech content based on your beliefs and strategies, and the speech content should not contain your thinking process.

    Now, please refer to your beliefs and predetermined strategies to generate your final speech. Start with a symbol: "#FINAL:".

    # BELIEF:
    {belief}

    # STRATEGY:
    {strategy}

    """
        return ret

    def prompt_vote(self, belief, strategy) -> str:
        """Generate voting prompt."""
        ret = f"""
    Your actions in each round are divided into four steps: 1 Analyze newly acquired information; 2. Update the identification of other players' identities; 3. Update your own strategy; 4. Decide on your own speech or action.
    Now it is step 4. Decide on your own action.

    Now you need to select a player for action (including voting, rescue, detect, etc.), you need to output the final goal in the format of "[X]", where X represents the player's ID and is a number. For example, "[1]" means you want to vote for player 1.

    Now, please refer to your beliefs and predetermined strategies to generate your final action goal. Start with a symbol: "#FINAL:".

    # BELIEF:
    {belief}

    # STRATEGY:
    {strategy}

    """
        return ret

    def parse_llm_response(self, response_text, tag_name):
        """Parse LLM response to extract content after specified tag."""
        index = response_text.find(tag_name)
        if index != -1:
            return response_text[index + len(tag_name):].strip()
        else:
            return response_text

    # Include strategy experience pool methods (same as EnhancedMichael)
    def _get_strategy_guidance(self, observation: str, phase: str) -> str:
        """Get strategic guidance from relevant past experiences."""
        try:
            matching_result = self.strategy_matcher.find_relevant_experiences(
                current_role=self.init_info.get("role", "Unknown"),
                current_phase=phase,
                current_round=self.round,
                observation=observation,
                agent_belief=self.belief,
                agent_strategy=self.strategy,
                max_results=3
            )

            guidance = self.strategy_matcher.format_strategy_guidance(matching_result)
            print(f"Found {len(matching_result.get('relevant_experiences', []))} relevant experiences")
            return guidance

        except Exception as e:
            print(f"Error getting strategy guidance: {e}")
            return ""

    def _enhanced_analyze(self, observation: str, strategy_guidance: str) -> str:
        """Enhanced analysis with experience guidance."""
        enhanced_prompt = f"""
{self.prompt_analyze(observation)}

# STRATEGIC EXPERIENCE GUIDANCE:
{strategy_guidance}

Please incorporate insights from relevant past experiences into your analysis.
"""

        return self.parse_llm_response(
            self.api(input_messages=[
                {"role": "system", "content": self.prompt_system()},
                {"role": "user", "content": enhanced_prompt}
            ]),
            "#SUMMARY:"
        )

    def _enhanced_strategy_update(self, analysis: str, belief: str, strategy_guidance: str) -> str:
        """Enhanced strategy update with experience guidance."""
        enhanced_prompt = f"""
{self.prompt_strategy(analysis, belief, self.strategy)}

# STRATEGIC EXPERIENCE GUIDANCE:
{strategy_guidance}

Please incorporate insights from relevant past experiences into your strategy development.
"""
        print("\n\n\n STRATEGY PROMPT \n" + enhanced_prompt)
        return self.parse_llm_response(
            self.api(input_messages=[
                {"role": "system", "content": self.prompt_system()},
                {"role": "user", "content": enhanced_prompt}
            ]),
            "#STRATEGY:"
        )

    def _evaluate_action_for_brilliance(self, observation: str, action: str, phase: str, round_number: int) -> None:
        """Evaluate the current action for potential brilliance."""
        try:
            # Create agent state for evaluation
            agent_state = {
                "belief": self.belief,
                "strategy": self.strategy,
                "round": self.round,
                "is_initialized": self.is_initialized,
                "init_info": self.init_info
            }

            # Create game context
            game_context = {
                "agent_role": self.init_info.get("role", "Unknown"),
                "phase": phase,
                "round": round_number
            }

            # Evaluate action for brilliance
            action_key = self.brilliant_action_detector.evaluate_action(
                player_id=self.init_info.get("player_id", 0),
                observation=observation,
                action=action,
                agent_state=agent_state,
                game_context=game_context
            )

            if action_key:
                self.pending_action_keys.append(action_key)
                print(f"Action submitted for brilliance evaluation: {action_key}")

        except Exception as e:
            print(f"Error evaluating action for brilliance: {e}")

    def _complete_pending_evaluations(self, new_observation: str) -> None:
        """Complete evaluation of pending actions using new observation."""
        completed_keys = []

        for action_key in self.pending_action_keys:
            try:
                experience_id = self.brilliant_action_detector.complete_action_evaluation(
                    action_key, new_observation
                )
                if experience_id:
                    print(f"Recorded brilliant action experience: {experience_id}")
                completed_keys.append(action_key)
            except Exception as e:
                print(f"Error completing action evaluation: {e}")
                completed_keys.append(action_key)

        # Remove completed evaluations
        for key in completed_keys:
            if key in self.pending_action_keys:
                self.pending_action_keys.remove(key)

    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get statistics about the learning process."""
        if not self.enable_learning:
            return {"learning_enabled": False}

        pool_stats = self.experience_pool.get_statistics()
        pending_count = self.brilliant_action_detector.get_pending_actions_count()

        return {
            "learning_enabled": True,
            "pool_statistics": pool_stats,
            "pending_evaluations": pending_count,
            "total_experiences": pool_stats["total_experiences"],
            "success_rate": pool_stats["success_rate"]
        }