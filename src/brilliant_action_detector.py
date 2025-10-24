"""
Brilliant Action Detector for Strategy Experience Pool
Identifies and analyzes brilliant strategic actions in Secret Mafia games.
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from .strategy_pool import StrategyExperiencePool
from .game_state_analyzer import GameStateAnalyzer


class BrilliantActionDetector:
    """
    Detects brilliant strategic actions and analyzes them for experience storage.
    Integrates with the game logger to capture exceptional gameplay moments.
    """

    def __init__(self,
                 experience_pool: StrategyExperiencePool,
                 state_analyzer: GameStateAnalyzer,
                 api_client,
                 model_name: str = "qwen3-8b"):
        """
        Initialize the brilliant action detector.

        Args:
            experience_pool: Strategy experience pool instance
            state_analyzer: Game state analyzer instance
            api_client: LLM API client
            model_name: Name of the LLM model to use
        """
        self.experience_pool = experience_pool
        self.state_analyzer = state_analyzer
        self.api_client = api_client
        self.model_name = model_name
        self.pending_actions = {}  # Store actions for delayed evaluation

    def evaluate_action(self,
                       player_id: int,
                       observation: str,
                       action: str,
                       agent_state: Dict[str, Any],
                       game_context: Dict[str, Any]) -> Optional[str]:
        """
        Evaluate an action to determine if it should be recorded as brilliant.

        Args:
            player_id: ID of the player who took the action
            observation: Game observation before the action
            action: Action taken by the agent
            agent_state: Internal state of the agent
            game_context: Game context information

        Returns:
            Experience ID if action was recorded as brilliant, None otherwise
        """
        # Skip first night as per requirement
        phase = game_context.get("phase", "")
        round_number = game_context.get("round", 1)
        if phase == "night" and round_number == 1:
            return None

        # Store action for delayed evaluation (we need next observation to assess outcome)
        action_key = f"{player_id}_{datetime.now().timestamp()}"
        self.pending_actions[action_key] = {
            "player_id": player_id,
            "observation": observation,
            "action": action,
            "agent_state": agent_state,
            "game_context": game_context,
            "timestamp": datetime.now().isoformat()
        }

        return action_key

    def complete_action_evaluation(self,
                                  action_key: str,
                                  next_observation: str) -> Optional[str]:
        """
        Complete the evaluation of a pending action using the next observation.

        Args:
            action_key: Key of the pending action
            next_observation: Observation after the action was taken

        Returns:
            Experience ID if action was recorded as brilliant, None otherwise
        """
        if action_key not in self.pending_actions:
            return None

        pending_action = self.pending_actions.pop(action_key)

        try:
            # Analyze the action outcome
            outcome_analysis = self.state_analyzer.analyze_action_outcome(
                action=pending_action["action"],
                action_type=self._classify_action_type(pending_action["action"], pending_action["game_context"]),
                previous_state=pending_action["observation"],
                next_observation=next_observation,
                agent_role=pending_action["game_context"].get("agent_role", "Unknown")
            )

            # Determine if this is a brilliant action
            if self._is_brilliant_action(outcome_analysis, pending_action):
                experience_id = self._create_experience_record(pending_action, next_observation, outcome_analysis)
                return experience_id

        except Exception as e:
            print(f"Error completing action evaluation: {e}")

        return None

    def _classify_action_type(self, action: str, game_context: Dict[str, Any]) -> str:
        """Classify the type of action taken."""
        if not action:
            return "unknown"

        action_lower = action.lower()

        # Check for voting action
        if re.match(r'^\[\d+\]$', action.strip()):
            return "vote"

        # Check for action keywords
        if any(word in action_lower for word in ["investigate", "detect", "check"]):
            return "investigate"
        elif any(word in action_lower for word in ["protect", "save", "heal"]):
            return "protect"
        elif any(word in action_lower for word in ["kill", "eliminate", "attack"]):
            return "attack"
        elif game_context.get("phase") == "night":
            return "night_action"
        elif len(action) > 10:  # Assume speech if substantial content
            return "speak"
        else:
            return "other"

    def _is_brilliant_action(self, outcome_analysis: Dict[str, Any], pending_action: Dict[str, Any]) -> bool:
        """
        Determine if an action qualifies as brilliant based on outcome analysis.

        Args:
            outcome_analysis: Analysis of the action outcome
            pending_action: The pending action data

        Returns:
            True if the action should be recorded as brilliant
        """
        # Check if explicitly marked as brilliant
        if outcome_analysis.get("is_brilliant", False):
            return True

        # Check for high success rating
        success_rating = outcome_analysis.get("success_rating", 3)
        if success_rating >= 4:
            return True

        # Check for high strategic value
        strategic_value = outcome_analysis.get("strategic_value", "medium")
        if strategic_value in ["high", "critical"]:
            return True

        # Check for specific brilliant patterns
        action = pending_action["action"]
        agent_role = pending_action["game_context"].get("agent_role", "")
        phase = pending_action["game_context"].get("phase", "")

        # Mafia actions that might be brilliant
        if agent_role == "Mafia" and phase == "night":
            if self._is_mafia_night_brilliant(action, pending_action):
                return True

        # Doctor actions that might be brilliant
        if agent_role == "Doctor" and phase == "night":
            if self._is_doctor_night_brilliant(action, pending_action, outcome_analysis):
                return True

        # Detective actions that might be brilliant
        if agent_role == "Detective" and phase == "night":
            if self._is_detective_night_brilliant(action, pending_action, outcome_analysis):
                return True

        # Villager actions that might be brilliant
        if agent_role in ["A regular villager", "Villager"] and phase.startswith("day"):
            if self._is_villager_day_brilliant(action, pending_action, outcome_analysis):
                return True

        return False

    def _is_mafia_night_brilliant(self, action: str, pending_action: Dict[str, Any]) -> bool:
        """Check if Mafia night action was brilliant."""
        # Check if they eliminated a key player (detective/doctor)
        observation = pending_action["observation"].lower()

        # Pattern: Mafia successfully identifies and eliminates detective/doctor
        if ("detective" in observation or "doctor" in observation) and self._action_suggests_target_selection(action):
            return True

        return False

    def _is_doctor_night_brilliant(self, action: str, pending_action: Dict[str, Any], outcome_analysis: Dict[str, Any]) -> bool:
        """Check if Doctor night action was brilliant."""
        impact = outcome_analysis.get("impact_description", "").lower()

        # Pattern: Doctor successfully saves someone from Mafia attack
        if any(word in impact for word in ["saved", "prevented", "blocked", "protected"]):
            return True

        # Pattern: Doctor protects someone who claims to be detective
        observation = pending_action["observation"].lower()
        if "detective" in observation and "claim" in observation:
            return True

        return False

    def _is_detective_night_brilliant(self, action: str, pending_action: Dict[str, Any], outcome_analysis: Dict[str, Any]) -> bool:
        """Check if Detective night action was brilliant."""
        impact = outcome_analysis.get("impact_description", "").lower()

        # Pattern: Detective successfully identifies a Mafia member
        if any(word in impact for word in ["mafia", "killer", "found", "identified", "discovered"]):
            return True

        return False

    def _is_villager_day_brilliant(self, action: str, pending_action: Dict[str, Any], outcome_analysis: Dict[str, Any]) -> bool:
        """Check if Villager day action was brilliant."""
        impact = outcome_analysis.get("impact_description", "").lower()

        # Pattern: Villager successfully convinces others to vote for Mafia
        if any(word in impact for word in ["eliminated", "voted out", "mafia removed", "correct vote"]):
            return True

        # Pattern: Villager makes a convincing defense when accused
        observation = pending_action["observation"].lower()
        if "accuse" in observation and any(word in impact for word in ["survived", "not eliminated", "voting avoided"]):
            return True

        return False

    def _action_suggests_target_selection(self, action: str) -> bool:
        """Check if action suggests strategic target selection."""
        action_lower = action.lower()
        strategic_indicators = [
            "detective", "doctor", "threat", "dangerous", "important",
            "target", "eliminate", "priority"
        ]
        return any(indicator in action_lower for indicator in strategic_indicators)

    def _create_experience_record(self,
                                 pending_action: Dict[str, Any],
                                 next_observation: str,
                                 outcome_analysis: Dict[str, Any]) -> str:
        """
        Create a detailed experience record for the brilliant action.

        Args:
            pending_action: The pending action data
            next_observation: Observation after the action
            outcome_analysis: Analysis of the action outcome

        Returns:
            Experience ID of the created record
        """
        # Analyze the state before the action
        state_analysis = self.state_analyzer.analyze_current_state(
            observation=pending_action["observation"],
            agent_role=pending_action["game_context"].get("agent_role", "Unknown"),
            phase=pending_action["game_context"].get("phase", "unknown"),
            round_number=pending_action["game_context"].get("round", 1),
            agent_belief=pending_action["agent_state"].get("belief", ""),
            agent_strategy=pending_action["agent_state"].get("strategy", "")
        )

        # Extract information from agent state
        agent_state = pending_action["agent_state"]
        game_context = pending_action["game_context"]

        # Create the experience record
        experience_id = self.experience_pool.add_experience(
            role=game_context.get("agent_role", "Unknown"),
            phase=game_context.get("phase", "unknown"),
            round_number=game_context.get("round", 1),
            key_info=state_analysis.get("key_info", ""),
            action=pending_action["action"],
            action_type=self._classify_action_type(pending_action["action"], game_context),
            reasoning=agent_state.get("strategy", ""),
            situation_assessment=state_analysis.get("parsed_analysis", {}).get("situation_type", "unknown"),
            strategic_thinking=agent_state.get("strategy", ""),
            key_factors=state_analysis.get("parsed_analysis", {}).get("critical_factors", []),
            success=outcome_analysis.get("success_rating", 3) >= 3,
            impact=outcome_analysis.get("impact_description", ""),
            additional_context={
                "urgency_level": state_analysis.get("urgency_level", "medium"),
                "strategic_importance": state_analysis.get("strategic_importance", "medium"),
                "success_rating": outcome_analysis.get("success_rating", 3),
                "strategic_value": outcome_analysis.get("strategic_value", "medium"),
                "observation_before": pending_action["observation"],
                "observation_after": next_observation,
                "agent_belief": agent_state.get("belief", ""),
                "brilliant_reasoning": outcome_analysis.get("strategic_reasoning", "")
            }
        )

        print(f"Created brilliant action experience: {experience_id}")
        return experience_id

    def get_pending_actions_count(self) -> int:
        """Get the number of pending actions awaiting evaluation."""
        return len(self.pending_actions)

    def clear_pending_actions(self) -> None:
        """Clear all pending actions (useful for game resets)."""
        self.pending_actions.clear()
        print("Cleared all pending brilliant action evaluations")

    def manual_record_brilliant_action(self,
                                      role: str,
                                      phase: str,
                                      round_number: int,
                                      key_info: str,
                                      action: str,
                                      reasoning: str,
                                      impact: str,
                                      success: bool = True) -> Optional[str]:
        """
        Manually record a brilliant action (for testing or manual annotation).

        Args:
            role: Agent role
            phase: Game phase
            round_number: Round number
            key_info: Key information in the situation
            action: Action taken
            reasoning: Reasoning behind the action
            impact: Impact of the action
            success: Whether the action was successful

        Returns:
            Experience ID if recorded successfully
        """
        return self.experience_pool.add_experience(
            role=role,
            phase=phase,
            round_number=round_number,
            key_info=key_info,
            action=action,
            action_type=self._classify_action_type(action, {"phase": phase}),
            reasoning=reasoning,
            situation_assessment="manually_recorded",
            strategic_thinking=reasoning,
            key_factors=[key_info],
            success=success,
            impact=impact
        )