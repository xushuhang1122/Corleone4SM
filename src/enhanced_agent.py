"""
Enhanced Agent with Strategy Experience Pool Integration
Integrates brilliant action learning into existing agent architecture.
"""

import json
import re
import time
from typing import Dict, List, Any, Optional
from xushuhang.agents.family import Michael
from .strategy_pool import StrategyExperiencePool
from .game_state_analyzer import GameStateAnalyzer
from .brilliant_action_detector import BrilliantActionDetector
from .strategy_matcher import StrategyMatcher


class EnhancedMichael(Michael):
    """
    Enhanced Michael agent with strategy experience pool integration.
    Learns from brilliant actions and uses past experiences to improve strategy.
    """

    def __init__(self, model_name: str, enable_learning: bool = True):
        """
        Initialize Enhanced Michael with experience pool capabilities.

        Args:
            model_name: Name of the LLM model to use
            enable_learning: Whether to enable learning from experiences
        """
        super().__init__(model_name)

        self.enable_learning = enable_learning

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

            print("Enhanced Michael initialized with strategy experience pool")
        else:
            print("Enhanced Michael initialized without learning (learning disabled)")

    def __call__(self, observation: str) -> str:
        """
        Enhanced call method with experience pool integration.

        Args:
            observation: Current game observation

        Returns:
            Agent action/response
        """
        print("\n=================================== ENHANCED MICHAEL ====================================")
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

    def manual_add_experience(self,
                             role: str,
                             phase: str,
                             round_number: int,
                             key_info: str,
                             action: str,
                             reasoning: str,
                             impact: str,
                             success: bool = True) -> Optional[str]:
        """Manually add a brilliant action experience."""
        if not self.enable_learning:
            print("Learning is disabled - cannot add experience")
            return None

        return self.brilliant_action_detector.manual_record_brilliant_action(
            role=role,
            phase=phase,
            round_number=round_number,
            key_info=key_info,
            action=action,
            reasoning=reasoning,
            impact=impact,
            success=success
        )

    def clear_learning_data(self) -> None:
        """Clear all learning data."""
        if self.enable_learning:
            self.experience_pool.clear_pool()
            self.brilliant_action_detector.clear_pending_actions()
            self.pending_action_keys.clear()
            print("All learning data cleared")


class EnhancedLoggedAgent:
    """
    Wrapper that adds both logging and experience pool capabilities to any agent.
    """

    def __init__(self, base_agent: Any, logger: Any, player_id: int, enable_learning: bool = True):
        """
        Initialize enhanced logged agent.

        Args:
            base_agent: The base agent instance
            logger: Game logger instance
            player_id: Player ID
            enable_learning: Whether to enable experience learning
        """
        self.base_agent = base_agent
        self.logger = logger
        self.player_id = player_id

        # Add experience pool capabilities if the base agent doesn't have them
        if enable_learning and not hasattr(base_agent, 'experience_pool'):
            self._add_experience_pool_capabilities()

    def _add_experience_pool_capabilities(self) -> None:
        """Add experience pool capabilities to the base agent."""
        if hasattr(self.base_agent, 'model_name'):
            # Create experience pool components
            self.base_agent.experience_pool = StrategyExperiencePool()
            self.base_agent.state_analyzer = GameStateAnalyzer(
                getattr(self.base_agent, 'api', None),
                self.base_agent.model_name
            )
            self.base_agent.brilliant_action_detector = BrilliantActionDetector(
                self.base_agent.experience_pool,
                self.base_agent.state_analyzer,
                getattr(self.base_agent, 'api', None),
                self.base_agent.model_name
            )
            self.base_agent.strategy_matcher = StrategyMatcher(
                self.base_agent.experience_pool,
                self.base_agent.state_analyzer,
                getattr(self.base_agent, 'api', None),
                self.base_agent.model_name
            )

            # Initialize tracking variables
            self.base_agent.previous_observation = None
            self.base_agent.pending_action_keys = []

    def __call__(self, observation: str) -> str:
        """Forward call with logging and experience tracking."""
        # Get agent state for logging
        agent_state = self._extract_agent_state()
        game_phase = self._extract_game_phase_from_agent()
        round_number = self._extract_round_from_agent()

        # Handle experience evaluation if applicable
        if (hasattr(self.base_agent, 'previous_observation') and
            self.base_agent.previous_observation and
            hasattr(self.base_agent, '_complete_pending_evaluations')):
            self.base_agent._complete_pending_evaluations(observation)

        # Get action from base agent
        action = self.base_agent(observation)

        # Log the turn
        self.logger.log_turn(
            player_id=self.player_id,
            observation=observation,
            agent_state=agent_state,
            action=action,
            game_phase=game_phase,
            round_number=round_number
        )

        # Evaluate action for brilliance if applicable
        if hasattr(self.base_agent, '_evaluate_action_for_brilliance'):
            self.base_agent._evaluate_action_for_brilliance(
                observation, action, game_phase or "unknown", round_number or 1
            )

        # Store observation for next cycle
        if hasattr(self.base_agent, 'previous_observation'):
            self.base_agent.previous_observation = observation

        return action

    def _extract_agent_state(self) -> Dict[str, Any]:
        """Extract agent state for logging."""
        state = {}
        if hasattr(self.base_agent, 'belief'):
            state['belief'] = getattr(self.base_agent, 'belief')
        if hasattr(self.base_agent, 'strategy'):
            state['strategy'] = getattr(self.base_agent, 'strategy')
        if hasattr(self.base_agent, 'round'):
            state['round'] = getattr(self.base_agent, 'round')
        if hasattr(self.base_agent, 'observation_history'):
            state['history_length'] = len(getattr(self.base_agent, 'observation_history', []))
        if hasattr(self.base_agent, 'init_info'):
            state['init_info'] = getattr(self.base_agent, 'init_info')
        if hasattr(self.base_agent, 'is_initialized'):
            state['is_initialized'] = getattr(self.base_agent, 'is_initialized')
        state['agent_class'] = self.base_agent.__class__.__name__
        return state

    def _extract_game_phase_from_agent(self) -> Optional[str]:
        """Extract game phase from agent."""
        if hasattr(self.base_agent, '_extract_game_phase_from_agent'):
            return self.base_agent._extract_game_phase_from_agent()
        return None

    def _extract_round_from_agent(self) -> Optional[int]:
        """Extract round number from agent."""
        if hasattr(self.base_agent, 'round'):
            return getattr(self.base_agent, 'round')
        return None

    def __getattr__(self, name):
        """Delegate attribute access to the base agent."""
        return getattr(self.base_agent, name)