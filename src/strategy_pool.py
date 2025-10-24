"""
Strategy Experience Pool for Mind Games Challenge
Manages storage and retrieval of strategic game experiences for Agent enhancement.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import re


class StrategyExperiencePool:
    """
    Manages a JSON-based pool of strategic game experiences for Agent learning.
    Stores brilliant actions and their contexts for future strategy matching.
    """

    def __init__(self, pool_file: str = "strategy_experiences/experiences.json"):
        """
        Initialize the strategy experience pool.

        Args:
            pool_file: Path to the JSON file storing experiences
        """
        self.pool_file = Path(pool_file)
        self.pool_file.parent.mkdir(parents=True, exist_ok=True)
        self.experiences = []
        self.index = {
            "by_role": {},
            "by_phase": {},
            "by_situation": {}
        }
        self.load_pool()

    def load_pool(self) -> None:
        """Load experiences from the JSON file."""
        try:
            if self.pool_file.exists():
                with open(self.pool_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.experiences = data.get("experiences", [])
                    self.index = data.get("index", {
                        "by_role": {},
                        "by_phase": {},
                        "by_situation": {}
                    })
                print(f"Loaded {len(self.experiences)} experiences from strategy pool")
            else:
                self._initialize_pool_file()
        except Exception as e:
            print(f"Error loading strategy pool: {e}")
            self._initialize_pool_file()

    def _initialize_pool_file(self) -> None:
        """Initialize a new pool file with basic structure."""
        initial_data = {
            "metadata": {
                "version": "1.0",
                "created": datetime.now().isoformat(),
                "description": "Strategy experience pool for Secret Mafia game brilliant actions",
                "total_experiences": 0
            },
            "experiences": [],
            "index": {
                "by_role": {},
                "by_phase": {},
                "by_situation": {},
                "last_updated": datetime.now().isoformat()
            }
        }

        with open(self.pool_file, 'w', encoding='utf-8') as f:
            json.dump(initial_data, f, indent=2, ensure_ascii=False)

    def add_experience(self,
                      role: str,
                      phase: str,
                      round_number: int,
                      key_info: str,
                      action: str,
                      action_type: str,
                      reasoning: str,
                      situation_assessment: str,
                      strategic_thinking: str,
                      key_factors: List[str],
                      success: bool,
                      impact: str,
                      additional_context: Optional[Dict] = None) -> str:
        """
        Add a new brilliant action experience to the pool.

        Args:
            role: Agent role (Mafia, Doctor, Detective, Villager)
            phase: Game phase (night, day_speak, day_vote)
            round_number: Current round number
            key_info: Key information available in this situation
            action: Action taken by the agent
            action_type: Type of action (vote, protect, investigate, speak, etc.)
            reasoning: Agent's reasoning for the action
            situation_assessment: LLM analysis of the situation
            strategic_thinking: Strategic considerations behind the action
            key_factors: Key factors that influenced the decision
            success: Whether the action was successful
            impact: Impact/effect of the action
            additional_context: Additional context information

        Returns:
            Unique ID of the added experience
        """
        # Skip first night as per requirement
        if phase == "night" and round_number == 1:
            print("Skipping first night experience as per rule")
            return None

        experience_id = str(uuid.uuid4())

        experience = {
            "experience_id": experience_id,
            "timestamp": datetime.now().isoformat(),
            "game_context": {
                "role": role,
                "phase": phase,
                "round": round_number,
                "key_info": key_info
            },
            "brilliant_action": {
                "action": action,
                "action_type": action_type,
                "reasoning": reasoning
            },
            "strategy_analysis": {
                "situation_assessment": situation_assessment,
                "strategic_thinking": strategic_thinking,
                "key_factors": key_factors
            },
            "outcome": {
                "success": success,
                "impact": impact
            },
            "additional_context": additional_context or {}
        }

        self.experiences.append(experience)
        self._update_index(experience)
        self.save_pool()

        print(f"Added experience {experience_id} for {role} in {phase}")
        return experience_id

    def _update_index(self, experience: Dict) -> None:
        """Update the search index with a new experience."""
        experience_id = experience["experience_id"]
        role = experience["game_context"]["role"]
        phase = experience["game_context"]["phase"]
        situation = experience["strategy_analysis"]["situation_assessment"]

        # Index by role
        if role not in self.index["by_role"]:
            self.index["by_role"][role] = []
        self.index["by_role"][role].append(experience_id)

        # Index by phase
        if phase not in self.index["by_phase"]:
            self.index["by_phase"][phase] = []
        self.index["by_phase"][phase].append(experience_id)

        # Index by situation (extract keywords)
        situation_keywords = self._extract_keywords(situation)
        for keyword in situation_keywords:
            if keyword not in self.index["by_situation"]:
                self.index["by_situation"][keyword] = []
            self.index["by_situation"][keyword].append(experience_id)

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for indexing."""
        # Simple keyword extraction - can be enhanced with more sophisticated NLP
        keywords = []

        # Common game-related keywords
        game_keywords = [
            "detective", "doctor", "mafia", "villager", "protect", "investigate",
            "vote", "accuse", "defend", "suspect", "claim", "reveal", "deceive",
            "critical", "important", "dangerous", "safe", "trust", "doubt"
        ]

        text_lower = text.lower()
        for keyword in game_keywords:
            if keyword in text_lower:
                keywords.append(keyword)

        return keywords

    def find_similar_experiences(self,
                                current_role: str,
                                current_phase: str,
                                current_key_info: str,
                                max_results: int = 5) -> List[Dict]:
        """
        Find experiences similar to the current game situation.

        Args:
            current_role: Current agent role
            current_phase: Current game phase
            current_key_info: Key information in current situation
            max_results: Maximum number of results to return

        Returns:
            List of similar experiences with relevance scores
        """
        if not self.experiences:
            return []

        # Filter by role and phase first
        candidate_ids = set()

        # Get candidates by role
        role_candidates = self.index["by_role"].get(current_role, [])
        candidate_ids.update(role_candidates)

        # Get candidates by phase
        phase_candidates = self.index["by_phase"].get(current_phase, [])
        candidate_ids.update(phase_candidates)

        # If no candidates by role/phase, return empty
        if not candidate_ids:
            return []

        # Convert to experiences and calculate relevance scores
        candidates = [exp for exp in self.experiences
                     if exp["experience_id"] in candidate_ids]

        # Score and rank candidates
        scored_candidates = []
        for candidate in candidates:
            score = self._calculate_similarity_score(
                current_key_info, candidate["game_context"]["key_info"]
            )
            scored_candidates.append((candidate, score))

        # Sort by score and return top results
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        return [candidate for candidate, _ in scored_candidates[:max_results]]

    def _calculate_similarity_score(self, info1: str, info2: str) -> float:
        """
        Calculate similarity score between two information strings.
        Simple implementation using keyword overlap.
        """
        keywords1 = set(self._extract_keywords(info1))
        keywords2 = set(self._extract_keywords(info2))

        if not keywords1 and not keywords2:
            return 0.5  # Neutral score for empty keyword sets

        if not keywords1 or not keywords2:
            return 0.1  # Low score for single-sided empty sets

        intersection = keywords1.intersection(keywords2)
        union = keywords1.union(keywords2)

        return len(intersection) / len(union) if union else 0.0

    def format_experiences_for_prompt(self, experiences: List[Dict]) -> str:
        """
        Format experiences for inclusion in agent strategy prompts.

        Args:
            experiences: List of experience dictionaries

        Returns:
            Formatted string suitable for prompt inclusion
        """
        if not experiences:
            return "No relevant past experiences found."

        formatted_lines = []
        formatted_lines.append("=== RELEVANT STRATEGIC EXPERIENCES ===")

        for i, exp in enumerate(experiences, 1):
            game_context = exp["game_context"]
            action = exp["brilliant_action"]
            analysis = exp["strategy_analysis"]
            outcome = exp["outcome"]

            experience_text = f"""
Experience {i}:
- Situation: {game_context['role']} in {game_context['phase']} phase
- Key Information: {game_context['key_info']}
- Action Taken: {action['action']} ({action['action_type']})
- Strategic Reasoning: {action['reasoning']}
- Strategic Analysis: {analysis['strategic_thinking']}
- Outcome: {outcome['impact']} (Success: {outcome['success']})
"""
            formatted_lines.append(experience_text)

        formatted_lines.append("=== END OF EXPERIENCES ===")
        return "\n".join(formatted_lines)

    def save_pool(self) -> None:
        """Save the current pool to file."""
        try:
            data = {
                "metadata": {
                    "version": "1.0",
                    "last_updated": datetime.now().isoformat(),
                    "description": "Strategy experience pool for Secret Mafia game brilliant actions",
                    "total_experiences": len(self.experiences)
                },
                "experiences": self.experiences,
                "index": {
                    "by_role": self.index["by_role"],
                    "by_phase": self.index["by_phase"],
                    "by_situation": self.index["by_situation"],
                    "last_updated": datetime.now().isoformat()
                }
            }

            with open(self.pool_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"Error saving strategy pool: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the experience pool."""
        stats = {
            "total_experiences": len(self.experiences),
            "experiences_by_role": {},
            "experiences_by_phase": {},
            "success_rate": 0.0
        }

        # Count by role and phase
        for exp in self.experiences:
            role = exp["game_context"]["role"]
            phase = exp["game_context"]["phase"]

            stats["experiences_by_role"][role] = stats["experiences_by_role"].get(role, 0) + 1
            stats["experiences_by_phase"][phase] = stats["experiences_by_phase"].get(phase, 0) + 1

        # Calculate success rate
        if self.experiences:
            successful = sum(1 for exp in self.experiences if exp["outcome"]["success"])
            stats["success_rate"] = successful / len(self.experiences)

        return stats

    def clear_pool(self) -> None:
        """Clear all experiences from the pool."""
        self.experiences = []
        self.index = {
            "by_role": {},
            "by_phase": {},
            "by_situation": {}
        }
        self.save_pool()
        print("Strategy pool cleared")