import re
import logging
from typing import Dict

EXPERT_METADATA = {
    "name": "semantic_router",
    "version": "1.0",
    "description": "Analyzes text to infer intent and route data to the correct Zenrube expert or flow.",
    "author": "vladinc@gmail.com"
}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticRouterExpert:
    """
    Expert class for analyzing input text, inferring intent, and determining
    the logical route (target expert or flow) without executing it.
    """

    def __init__(self):
        # Intent detection keywords
        self.intent_keywords = {
            "error": ["error", "bug", "failed", "crash", "exception"],
            "invoice": ["invoice", "bill", "payment", "receipt", "charge"],
            "meeting": ["meeting", "schedule", "appointment", "call", "conference"],
            "support": ["help", "support", "assistance", "question", "issue"],
            "urgent": ["urgent", "emergency", "asap", "critical", "priority"],
        }

        # Route mappings (static targets)
        self.route_mappings = {
            "error": "debug_expert",
            "invoice": "finance_handler",
            "meeting": "calendar_flow",
            "support": "support_agent",
            "urgent": "priority_handler",
            "unknown": "general_handler",
        }

    def run(self, input_data: str) -> Dict[str, str]:
        """
        Analyze the input text and return its inferred intent and routing target.

        Args:
            input_data (str): Plain text input.

        Returns:
            dict: {
                "input": original text,
                "intent": inferred intent,
                "route": target expert or flow
            }
        """
        if not isinstance(input_data, str):
            logger.warning("Non-string input received, converting to string.")
            input_data = str(input_data)

        text_preview = input_data[:50] + ("..." if len(input_data) > 50 else "")
        logger.info(f"Processing input: {text_preview}")

        intent = self._analyze_intent(input_data)
        route = self._route_to_target(intent)

        result = {"input": input_data, "intent": intent, "route": route}
        logger.info(f"Inferred intent='{intent}' → route='{route}'")
        return result

    def _analyze_intent(self, text: str) -> str:
        """
        Infer intent based on simple keyword matching.
        """
        text_lower = text.lower()
        for intent, keywords in self.intent_keywords.items():
            for keyword in keywords:
                if re.search(r"\b" + re.escape(keyword) + r"\b", text_lower):
                    logger.debug(f"Matched intent '{intent}' for keyword '{keyword}'")
                    return intent
        return "unknown"

    def _route_to_target(self, intent: str) -> str:
        """
        Map an intent to its corresponding routing target.
        """
        return self.route_mappings.get(intent, self.route_mappings["unknown"])
