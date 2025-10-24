"""
Strategy Matcher for Strategy Experience Pool
Intelligent matching of current game situations to relevant past experiences.
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from .strategy_pool import StrategyExperiencePool
from .game_state_analyzer import GameStateAnalyzer


class StrategyMatcher:
    """
    Intelligently matches current game situations to relevant past experiences
    and provides strategic guidance to agents.
    """

    def __init__(self,
                 experience_pool: StrategyExperiencePool,
                 state_analyzer: GameStateAnalyzer,
                 api_client,
                 model_name: str = "qwen3-8b"):
        """
        Initialize the strategy matcher.

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

    def find_relevant_experiences(self,
                                 current_role: str,
                                 current_phase: str,
                                 current_round: int,
                                 observation: str,
                                 agent_belief: str,
                                 agent_strategy: str,
                                 max_results: int = 3) -> Dict[str, Any]:
        """
        Find and rank relevant experiences for the current situation.

        Args:
            current_role: Current agent role
            current_phase: Current game phase
            current_round: Current round number
            observation: Current game observation
            agent_belief: Agent's current beliefs
            agent_strategy: Agent's current strategy
            max_results: Maximum number of experiences to return

        Returns:
            Dictionary containing relevant experiences and matching analysis
        """
        # First analyze the current state
        state_analysis = self.state_analyzer.analyze_current_state(
            observation=observation,
            agent_role=current_role,
            phase=current_phase,
            round_number=current_round,
            agent_belief=agent_belief,
            agent_strategy=agent_strategy
        )

        # Find similar experiences using the pool
        similar_experiences = self.experience_pool.find_similar_experiences(
            current_role=current_role,
            current_phase=current_phase,
            current_key_info=state_analysis.get("key_info", ""),
            max_results=max_results * 2  # Get more for semantic matching
        )

        # Use LLM to perform semantic matching and ranking
        ranked_experiences = self._semantic_rank_experiences(
            current_observation=observation,
            current_state=state_analysis,
            candidate_experiences=similar_experiences,
            max_results=max_results
        )

        return {
            "current_state_analysis": state_analysis,
            "relevant_experiences": ranked_experiences,
            "matching_summary": self._generate_matching_summary(state_analysis, ranked_experiences),
            "recommendations": self._generate_recommendations(ranked_experiences, current_role, current_phase)
        }

    def _semantic_rank_experiences(self,
                                  current_observation: str,
                                  current_state: Dict[str, Any],
                                  candidate_experiences: List[Dict],
                                  max_results: int) -> List[Dict]:
        """
        Use LLM to semantically rank candidate experiences by relevance.

        Args:
            current_observation: Current game observation
            current_state: Analysis of current state
            candidate_experiences: List of candidate experiences
            max_results: Maximum results to return

        Returns:
            Ranked list of experiences with relevance scores
        """
        if not candidate_experiences:
            return []

        # Prepare the ranking prompt
        ranking_prompt = self._build_ranking_prompt(
            current_observation, current_state, candidate_experiences
        )

        try:
            response = self.api_client(
                input_messages=[
                    {"role": "system", "content": self._get_ranking_system_prompt()},
                    {"role": "user", "content": ranking_prompt}
                ],
                temperature=0.2,  # Low temperature for consistent ranking
                model=self.model_name,
                max_tokens=2048
            )

            ranking_text = response.strip()
            parsed_ranking = self._parse_ranking_response(ranking_text, candidate_experiences)

            return parsed_ranking[:max_results]

        except Exception as e:
            print(f"Error in semantic ranking: {e}")
            # Fallback to simple keyword-based ranking
            return self._fallback_keyword_ranking(current_state, candidate_experiences, max_results)

    def _get_ranking_system_prompt(self) -> str:
        """Get the system prompt for experience ranking."""
        return """You are an expert Secret Mafia strategist tasked with finding the most relevant past experiences for current game situations.

Your task is to:
1. Analyze the current game situation
2. Compare it with candidate past experiences
3. Rank experiences by strategic relevance
4. Provide scores (0-10) and reasoning

Focus on strategic similarity rather than exact matching. Consider:
- Similar game situations and challenges
- Comparable strategic dilemmas
- Relevant role-specific considerations
- Similar game phases and contexts

Return rankings in the specified format with clear reasoning for each score."""

    def _build_ranking_prompt(self,
                             current_observation: str,
                             current_state: Dict[str, Any],
                             candidate_experiences: List[Dict]) -> str:
        """Build the prompt for experience ranking."""
        current_info = current_state.get("key_info", "")
        situation_type = current_state.get("parsed_analysis", {}).get("situation_type", "unknown")
        urgency = current_state.get("urgency_level", "medium")

        prompt_parts = [
            f"CURRENT SITUATION:",
            f"- Observation: {current_observation}",
            f"- Key Information: {current_info}",
            f"- Situation Type: {situation_type}",
            f"- Urgency Level: {urgency}",
            "",
            "PAST EXPERIENCES TO EVALUATE:"
        ]

        for i, exp in enumerate(candidate_experiences, 1):
            game_context = exp["game_context"]
            action = exp["brilliant_action"]
            analysis = exp["strategy_analysis"]
            outcome = exp["outcome"]

            experience_text = f"""
Experience {i}:
- Situation: {game_context['role']} in {game_context['phase']} phase (Round {game_context['round']})
- Key Information: {game_context['key_info']}
- Action Taken: {action['action']} ({action['action_type']})
- Strategic Reasoning: {action['reasoning']}
- Strategic Analysis: {analysis['strategic_thinking']}
- Outcome: {outcome['impact']} (Success: {outcome['success']})
"""
            prompt_parts.append(experience_text)

        prompt_parts.extend([
            "",
            "Please rank these experiences by relevance to the current situation.",
            "Use this format:",
            "",
            "RANKING:",
            "1. Experience [X] - Score: [0-10] - [Reasoning for relevance]",
            "2. Experience [Y] - Score: [0-10] - [Reasoning for relevance]",
            "3. Experience [Z] - Score: [0-10] - [Reasoning for relevance]",
            "",
            "Consider strategic similarity, contextual relevance, and learning value."
        ])

        return "\n".join(prompt_parts)

    def _parse_ranking_response(self, ranking_text: str, candidate_experiences: List[Dict]) -> List[Dict]:
        """Parse the ranking response and attach scores to experiences."""
        ranked_experiences = []

        # Extract rankings using regex
        ranking_pattern = r'(\d+)\.\s*Experience\s+(\d+)\s*-\s*Score:\s*(\d+(?:\.\d+)?)\s*-\s*(.+)'
        matches = re.findall(ranking_pattern, ranking_text, re.IGNORECASE | re.MULTILINE)

        for rank, exp_num, score, reasoning in matches:
            try:
                exp_index = int(exp_num) - 1  # Convert to 0-based index
                if 0 <= exp_index < len(candidate_experiences):
                    experience = candidate_experiences[exp_index].copy()
                    experience["relevance_score"] = float(score)
                    experience["relevance_reasoning"] = reasoning.strip()
                    experience["rank"] = int(rank)
                    ranked_experiences.append(experience)
            except (ValueError, IndexError):
                continue

        # If no matches found, return all experiences with neutral scores
        if not ranked_experiences:
            for i, exp in enumerate(candidate_experiences):
                exp_copy = exp.copy()
                exp_copy["relevance_score"] = 5.0  # Neutral score
                exp_copy["relevance_reasoning"] = "Unable to parse ranking - assigned neutral score"
                exp_copy["rank"] = i + 1
                ranked_experiences.append(exp_copy)

        # Sort by rank
        ranked_experiences.sort(key=lambda x: x.get("rank", float('inf')))
        return ranked_experiences

    def _fallback_keyword_ranking(self,
                                 current_state: Dict[str, Any],
                                 candidate_experiences: List[Dict],
                                 max_results: int) -> List[Dict]:
        """Fallback ranking using keyword similarity."""
        current_keywords = set()
        current_text = current_state.get("key_info", "").lower()

        # Extract keywords from current state
        game_keywords = [
            "detective", "doctor", "mafia", "villager", "protect", "investigate",
            "vote", "accuse", "defend", "suspect", "claim", "reveal", "deceive"
        ]
        current_keywords.update([kw for kw in game_keywords if kw in current_text])

        # Score experiences based on keyword overlap
        scored_experiences = []
        for exp in candidate_experiences:
            exp_text = exp["game_context"]["key_info"].lower()
            exp_keywords = set([kw for kw in game_keywords if kw in exp_text])

            # Calculate similarity score
            if current_keywords and exp_keywords:
                intersection = current_keywords.intersection(exp_keywords)
                union = current_keywords.union(exp_keywords)
                score = (len(intersection) / len(union)) * 10
            else:
                score = 5.0  # Neutral score

            exp_copy = exp.copy()
            exp_copy["relevance_score"] = score
            exp_copy["relevance_reasoning"] = f"Keyword similarity score: {score:.1f}/10"
            exp_copy["rank"] = 0
            scored_experiences.append(exp_copy)

        # Sort by score and return top results
        scored_experiences.sort(key=lambda x: x["relevance_score"], reverse=True)
        return scored_experiences[:max_results]

    def _generate_matching_summary(self, current_state: Dict[str, Any], ranked_experiences: List[Dict]) -> str:
        """Generate a summary of the matching results."""
        if not ranked_experiences:
            return "No relevant experiences found for the current situation."

        situation_type = current_state.get("parsed_analysis", {}).get("situation_type", "unknown")
        urgency = current_state.get("urgency_level", "medium")

        summary_parts = [
            f"Found {len(ranked_experiences)} relevant experiences for {situation_type} situation (urgency: {urgency}).",
            ""
        ]

        for i, exp in enumerate(ranked_experiences, 1):
            game_context = exp["game_context"]
            action = exp["brilliant_action"]
            score = exp.get("relevance_score", 0)
            reasoning = exp.get("relevance_reasoning", "")

            summary_parts.append(
                f"{i}. {game_context['role']} experience (Score: {score:.1f}/10): {action['action']}"
            )
            if reasoning:
                summary_parts.append(f"   Relevance: {reasoning}")

        return "\n".join(summary_parts)

    def _generate_recommendations(self, ranked_experiences: List[Dict], current_role: str, current_phase: str) -> List[str]:
        """Generate strategic recommendations based on relevant experiences."""
        recommendations = []

        if not ranked_experiences:
            return ["No specific recommendations available - proceed with standard strategy."]

        # Extract strategic insights from top experiences
        top_experience = ranked_experiences[0]
        strategic_thinking = top_experience.get("strategy_analysis", {}).get("strategic_thinking", "")
        action_reasoning = top_experience.get("brilliant_action", {}).get("reasoning", "")

        if strategic_thinking:
            recommendations.append(f"Key strategic insight: {strategic_thinking}")

        if action_reasoning:
            recommendations.append(f"Relevant action reasoning: {action_reasoning}")

        # Add role-specific recommendations
        if current_role == "Mafia":
            recommendations.extend(self._get_mafia_recommendations(ranked_experiences))
        elif current_role == "Doctor":
            recommendations.extend(self._get_doctor_recommendations(ranked_experiences))
        elif current_role == "Detective":
            recommendations.extend(self._get_detective_recommendations(ranked_experiences))
        else:
            recommendations.extend(self._get_villager_recommendations(ranked_experiences))

        # Add phase-specific recommendations
        if current_phase == "night":
            recommendations.extend(self._get_night_recommendations(ranked_experiences))
        elif current_phase == "day_vote":
            recommendations.extend(self._get_voting_recommendations(ranked_experiences))

        return recommendations[:5]  # Limit to top 5 recommendations

    def _get_mafia_recommendations(self, experiences: List[Dict]) -> List[str]:
        """Get Mafia-specific recommendations."""
        recommendations = []

        # Look for patterns in successful Mafia actions
        successful_mafia = [exp for exp in experiences
                           if exp["game_context"]["role"] == "Mafia" and exp["outcome"]["success"]]

        if successful_mafia:
            recommendations.append("Consider deception strategies that worked in similar situations.")
            recommendations.append("Focus on eliminating key threats (detective/doctor) when possible.")

        return recommendations

    def _get_doctor_recommendations(self, experiences: List[Dict]) -> List[str]:
        """Get Doctor-specific recommendations."""
        recommendations = []

        successful_doctor = [exp for exp in experiences
                            if exp["game_context"]["role"] == "Doctor" and exp["outcome"]["success"]]

        if successful_doctor:
            recommendations.append("Prioritize protecting players who claim important roles.")
            recommendations.append("Consider patterns in Mafia targeting to anticipate next victims.")

        return recommendations

    def _get_detective_recommendations(self, experiences: List[Dict]) -> List[str]:
        """Get Detective-specific recommendations."""
        recommendations = []

        successful_detective = [exp for exp in experiences
                              if exp["game_context"]["role"] == "Detective" and exp["outcome"]["success"]]

        if successful_detective:
            recommendations.append("Focus investigation on suspicious behavioral patterns.")
            recommendations.append("Consider timing of role reveals for maximum impact.")

        return recommendations

    def _get_villager_recommendations(self, experiences: List[Dict]) -> List[str]:
        """Get Villager-specific recommendations."""
        recommendations = []

        successful_villager = [exp for exp in experiences
                              if exp["game_context"]["role"] in ["A regular villager", "Villager"]
                              and exp["outcome"]["success"]]

        if successful_villager:
            recommendations.append("Pay attention to voting patterns and accusations.")
            recommendations.append("Support logical reasoning and evidence-based decisions.")

        return recommendations

    def _get_night_recommendations(self, experiences: List[Dict]) -> List[str]:
        """Get night phase recommendations."""
        return ["Night actions are crucial - prioritize targets carefully.",
                "Consider the information gained from previous day discussions."]

    def _get_voting_recommendations(self, experiences: List[Dict]) -> List[str]:
        """Get voting phase recommendations."""
        return ["Voting decisions should be based on accumulated evidence.",
                "Consider the implications of each potential elimination on team balance."]

    def format_strategy_guidance(self, matching_result: Dict[str, Any]) -> str:
        """
        Format concise strategy guidance for agent use.
        Removes detailed experiences and matching summary, focusing on distilled insights.

        Args:
            matching_result: Result from find_relevant_experiences

        Returns:
            Formatted guidance string suitable for agent prompts
        """
        guidance_parts = []

        # Add current situation analysis
        state_analysis = matching_result.get("current_state_analysis", {})
        key_info = state_analysis.get("key_info", "")
        situation_type = state_analysis.get("parsed_analysis", {}).get("situation_type", "unknown")
        urgency = state_analysis.get("urgency_level", "medium")
        critical_factors = state_analysis.get("critical_factors", [])

        guidance_parts.extend([
            "=== STRATEGIC SITUATION ANALYSIS ===",
            f"Current Situation Type: {situation_type}",
            f"Key Information: {key_info}",
            f"Urgency Level: {urgency}",
            ""
        ])

        # Add critical factors if available
        if critical_factors:
            guidance_parts.append("=== CRITICAL FACTORS ===")
            for factor in critical_factors:
                guidance_parts.append(f"• {factor}")
            guidance_parts.append("")

        # Extract and add experience-based insights without showing full experiences
        relevant_experiences = matching_result.get("relevant_experiences", [])
        if relevant_experiences:
            guidance_parts.append("=== EXPERIENCE-BASED INSIGHTS ===")

            # Add confidence metrics from experiences
            exp_count = len(relevant_experiences)
            avg_relevance = sum(exp.get("relevance_score", 0) for exp in relevant_experiences) / exp_count if exp_count > 0 else 0
            success_count = sum(1 for exp in relevant_experiences if exp["outcome"]["success"])
            success_rate = (success_count / exp_count * 100) if exp_count > 0 else 0

            guidance_parts.extend([
                f"Based on {exp_count} similar experiences (avg relevance: {avg_relevance:.1f}/10)",
                f"Success rate in similar situations: {success_rate:.0f}%",
                ""
            ])

        # Add distilled strategic recommendations
        recommendations = matching_result.get("recommendations", [])
        if recommendations:
            guidance_parts.extend([
                "=== STRATEGIC RECOMMENDATIONS ===",
                *[f"• {rec}" for rec in recommendations],
                ""
            ])

        return "\n".join(guidance_parts)