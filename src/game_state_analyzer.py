"""
Game State Analyzer for Strategy Experience Pool
Uses LLM to analyze game states and extract key strategic information.
"""

import json
import re
from typing import Dict, List, Any, Optional
from .strategy_pool import StrategyExperiencePool


class GameStateAnalyzer:
    """
    Analyzes game states using LLM to extract key strategic information
    for experience matching and storage.
    """

    def __init__(self, api_client, model_name: str = "qwen3-8b"):
        """
        Initialize the game state analyzer.

        Args:
            api_client: LLM API client (from agent classes)
            model_name: Name of the LLM model to use
        """
        self.api_client = api_client
        self.model_name = model_name

    def analyze_current_state(self,
                             observation: str,
                             agent_role: str,
                             phase: str,
                             round_number: int,
                             agent_belief: str,
                             agent_strategy: str) -> Dict[str, Any]:
        """
        Analyze the current game state and extract key information.

        Args:
            observation: Current game observation
            agent_role: Current agent role
            phase: Current game phase
            round_number: Current round number
            agent_belief: Agent's current beliefs
            agent_strategy: Agent's current strategy

        Returns:
            Dictionary containing analyzed state information
        """
        analysis_prompt = self._build_state_analysis_prompt(
            observation, agent_role, phase, round_number, agent_belief, agent_strategy
        )

        try:
            response = self.api_client(
                input_messages=[
                    {"role": "system", "content": self._get_analysis_system_prompt()},
                    {"role": "user", "content": analysis_prompt}
                ],
                temperature=0.3,
                model=self.model_name,
                max_tokens=1024
            )

            analysis_text = response.strip()
            parsed_analysis = self._parse_analysis_response(analysis_text)

            return {
                "raw_analysis": analysis_text,
                "parsed_analysis": parsed_analysis,
                "key_info": parsed_analysis.get("key_information", ""),
                "situation_type": parsed_analysis.get("situation_type", "unknown"),
                "critical_factors": parsed_analysis.get("critical_factors", []),
                "urgency_level": parsed_analysis.get("urgency_level", "medium"),
                "strategic_importance": parsed_analysis.get("strategic_importance", "medium")
            }

        except Exception as e:
            print(f"Error analyzing game state: {e}")
            return {
                "raw_analysis": "",
                "parsed_analysis": {},
                "key_info": self._extract_fallback_key_info(observation, agent_role),
                "situation_type": "unknown",
                "critical_factors": [],
                "urgency_level": "medium",
                "strategic_importance": "medium"
            }

    def _get_analysis_system_prompt(self) -> str:
        """Get the system prompt for game state analysis."""
        return """You are an expert Secret Mafia game analyst. Your task is to analyze game states and extract key strategic information.

For each game state analysis, provide:
1. Key Information: The most important facts in this situation
2. Situation Type: Categorize the situation (e.g., "being_accused", "investigating_suspects", "protecting_key_player", "deception_opportunity", "voting_critical")
3. Critical Factors: List the key factors influencing strategy
4. Urgency Level: How urgent is this situation (low/medium/high/critical)
5. Strategic Importance: How strategically important is this moment (low/medium/high/critical)

Focus on information that would be valuable for making strategic decisions in similar future situations."""

    def _build_state_analysis_prompt(self,
                                   observation: str,
                                   agent_role: str,
                                   phase: str,
                                   round_number: int,
                                   agent_belief: str,
                                   agent_strategy: str) -> str:
        """Build the prompt for state analysis."""
        return f"""Analyze this Secret Mafia game state and extract key strategic information.

Game Context:
- Your Role: {agent_role}
- Phase: {phase}
- Round: {round_number}
- Current Observation: {observation}

Your Current State:
- Beliefs: {agent_belief}
- Strategy: {agent_strategy}

Please provide a structured analysis in the following format:

KEY_INFORMATION: [Most important facts in this situation - what should an agent focus on?]

SITUATION_TYPE: [Categorize this situation type from: being_accused, investigating_suspects, protecting_key_player, deception_opportunity, voting_critical, routine_discussion, night_decision]

CRITICAL_FACTORS: [List 3-5 key factors that should influence strategy decisions]

URGENCY_LEVEL: [low/medium/high/critical - how urgent is action needed?]

STRATEGIC_IMPORTANCE: [low/medium/high/critical - how important is this moment for the game outcome?]

Focus on extracting information that would be most valuable for strategy learning and matching similar situations in the future."""

    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse the analysis response into structured data."""
        parsed = {}

        # Extract key information
        key_info_match = re.search(r'KEY_INFORMATION:\s*(.+?)(?=\n\S+:|$)', response, re.DOTALL | re.IGNORECASE)
        if key_info_match:
            parsed["key_information"] = key_info_match.group(1).strip()

        # Extract situation type
        situation_match = re.search(r'SITUATION_TYPE:\s*(.+?)(?=\n\S+:|$)', response, re.IGNORECASE)
        if situation_match:
            parsed["situation_type"] = situation_match.group(1).strip().lower()

        # Extract critical factors
        factors_match = re.search(r'CRITICAL_FACTORS:\s*(.+?)(?=\n\S+:|$)', response, re.DOTALL | re.IGNORECASE)
        if factors_match:
            factors_text = factors_match.group(1).strip()
            # Try to extract list items
            factor_list = re.findall(r'[-*]\s*(.+?)(?=\n[-*]|$)', factors_text, re.DOTALL)
            if factor_list:
                parsed["critical_factors"] = [factor.strip() for factor in factor_list]
            else:
                # Fallback: split by common delimiters
                parsed["critical_factors"] = [f.strip() for f in re.split(r'[,;]\s*', factors_text) if f.strip()]

        # Extract urgency level
        urgency_match = re.search(r'URGENCY_LEVEL:\s*(.+?)(?=\n\S+:|$)', response, re.IGNORECASE)
        if urgency_match:
            parsed["urgency_level"] = urgency_match.group(1).strip().lower()

        # Extract strategic importance
        importance_match = re.search(r'STRATEGIC_IMPORTANCE:\s*(.+?)(?=\n\S+:|$)', response, re.IGNORECASE)
        if importance_match:
            parsed["strategic_importance"] = importance_match.group(1).strip().lower()

        return parsed

    def _extract_fallback_key_info(self, observation: str, agent_role: str) -> str:
        """Extract basic key information without LLM as fallback."""
        key_info_parts = []

        # Extract role information
        if "detective" in observation.lower():
            key_info_parts.append("detective mentioned")
        if "doctor" in observation.lower():
            key_info_parts.append("doctor mentioned")
        if "mafia" in observation.lower():
            key_info_parts.append("mafia discussion")

        # Extract voting information
        vote_matches = re.findall(r'\[(\d+)\]', observation)
        if vote_matches:
            key_info_parts.append(f"voting activity detected: players {', '.join(vote_matches)}")

        # Extract accusations
        accusation_words = ["accuse", "suspect", "mafia", "killer"]
        if any(word in observation.lower() for word in accusation_words):
            key_info_parts.append("accusations or suspicions present")

        # Extract role claims
        claim_patterns = [
            r'i\s+am\s+(?:the\s+)?(detective|doctor)',
            r'(detective|doctor)\s+here',
            r'claim\s+(?:to\s+be\s+)?(?:the\s+)?(detective|doctor)'
        ]
        for pattern in claim_patterns:
            if re.search(pattern, observation.lower()):
                key_info_parts.append("role claim detected")
                break

        return "; ".join(key_info_parts) if key_info_parts else "basic game information available"

    def analyze_action_outcome(self,
                              action: str,
                              action_type: str,
                              previous_state: str,
                              next_observation: str,
                              agent_role: str) -> Dict[str, Any]:
        """
        Analyze the outcome of an action to determine if it was brilliant.

        Args:
            action: Action taken by the agent
            action_type: Type of action (vote, speak, protect, investigate)
            previous_state: State before the action
            next_observation: Observation after the action
            agent_role: Role of the agent who took the action

        Returns:
            Dictionary containing outcome analysis
        """
        outcome_prompt = self._build_outcome_analysis_prompt(
            action, action_type, previous_state, next_observation, agent_role
        )

        try:
            response = self.api_client(
                input_messages=[
                    {"role": "system", "content": self._get_outcome_analysis_system_prompt()},
                    {"role": "user", "content": outcome_prompt}
                ],
                temperature=0.3,
                model=self.model_name,
                max_tokens=1024
            )

            outcome_text = response.strip()
            parsed_outcome = self._parse_outcome_response(outcome_text)

            return {
                "raw_outcome_analysis": outcome_text,
                "parsed_outcome": parsed_outcome,
                "is_brilliant": parsed_outcome.get("is_brilliant", False),
                "success_rating": parsed_outcome.get("success_rating", 3),
                "impact_description": parsed_outcome.get("impact_description", ""),
                "strategic_value": parsed_outcome.get("strategic_value", "medium")
            }

        except Exception as e:
            print(f"Error analyzing action outcome: {e}")
            return {
                "raw_outcome_analysis": "",
                "parsed_outcome": {},
                "is_brilliant": False,
                "success_rating": 3,
                "impact_description": "Unable to analyze outcome",
                "strategic_value": "medium"
            }

    def _get_outcome_analysis_system_prompt(self) -> str:
        """Get the system prompt for outcome analysis."""
        return """You are an expert Secret Mafia game analyst evaluating the quality of actions.

Evaluate whether the action was "brilliant" - meaning it demonstrated exceptional strategic thinking, led to positive outcomes, or involved clever gameplay.

Consider:
- Strategic value and cleverness
- Effectiveness in achieving goals
- Impact on the game state
- Quality of reasoning and timing

Rate actions on success (1-5 scale) and identify truly brilliant plays that should be learned from."""

    def _build_outcome_analysis_prompt(self,
                                     action: str,
                                     action_type: str,
                                     previous_state: str,
                                     next_observation: str,
                                     agent_role: str) -> str:
        """Build the prompt for outcome analysis."""
        return f"""Evaluate this Secret Mafia action to determine if it was brilliant.

Agent Action Details:
- Role: {agent_role}
- Action: {action}
- Action Type: {action_type}

Context:
- State Before Action: {previous_state}
- Observation After Action: {next_observation}

Please evaluate using this format:

IS_BRILLIANT: [true/false - Was this an exceptionally clever/strategic action?]

SUCCESS_RATING: [1-5 - How successful was this action? 1=failed, 5=perfect execution]

IMPACT_DESCRIPTION: [Describe what happened as a result of this action]

STRATEGIC_VALUE: [low/medium/high/critical - How strategically valuable was this action?]

STRATEGIC_REASONING: [Why was this action good/bad from a strategic perspective?]

Focus on identifying actions that demonstrate exceptional gameplay that others could learn from."""

    def _parse_outcome_response(self, response: str) -> Dict[str, Any]:
        """Parse the outcome analysis response."""
        parsed = {}

        # Extract is_brilliant
        brilliant_match = re.search(r'IS_BRILLIANT:\s*(.+?)(?=\n\S+:|$)', response, re.IGNORECASE)
        if brilliant_match:
            brilliant_text = brilliant_match.group(1).strip().lower()
            parsed["is_brilliant"] = "true" in brilliant_text

        # Extract success rating
        rating_match = re.search(r'SUCCESS_RATING:\s*(\d+)', response, re.IGNORECASE)
        if rating_match:
            parsed["success_rating"] = int(rating_match.group(1))

        # Extract impact description
        impact_match = re.search(r'IMPACT_DESCRIPTION:\s*(.+?)(?=\n\S+:|$)', response, re.DOTALL | re.IGNORECASE)
        if impact_match:
            parsed["impact_description"] = impact_match.group(1).strip()

        # Extract strategic value
        value_match = re.search(r'STRATEGIC_VALUE:\s*(.+?)(?=\n\S+:|$)', response, re.IGNORECASE)
        if value_match:
            parsed["strategic_value"] = value_match.group(1).strip().lower()

        # Extract strategic reasoning
        reasoning_match = re.search(r'STRATEGIC_REASONING:\s*(.+?)(?=\n\S+:|$)', response, re.DOTALL | re.IGNORECASE)
        if reasoning_match:
            parsed["strategic_reasoning"] = reasoning_match.group(1).strip()

        return parsed