import re
import logging
import math
from typing import Dict, Optional

# Optional embeddings support
try:
    from zenrube.embeddings.client import embed_text
    from zenrube.embeddings.index import search
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logging.info("Embeddings module not available, routing will use keyword matching only")

EXPERT_METADATA = {
    "name": "semantic_router",
    "version": "1.0",
    "description": "Analyzes text to infer intent and route data to the correct Zenrube expert or flow.",
    "author": "vladinc@gmail.com"
}




# Experts available with optional embedding namespaces
EXPERTS = {
    "systems_architect": {
        "keywords": [
            "architecture", "system", "scalability",
            "design", "blueprint", "refactor",
            "diagnostic", "root cause", "infrastructure",
            "performance", "bottleneck"
        ],
        "embedding_namespace": "systems_architect"
    },
    "security_analyst": {
        "keywords": [
            "security", "vulnerability", "breach", "risk",
            "attack", "exploit", "penetration", "secure"
        ],
        "embedding_namespace": "security_analyst"
    },
    "summarizer": {
        "keywords": [
            "summarize", "summary", "condense",
            "tl;dr", "shorten"
        ],
        "embedding_namespace": "summarizer"
    },
    "pragmatic_engineer": {
        "keywords": [
            "implement", "fix", "code", "debug",
            "practical", "engineer", "solution"
        ]
    },
    "publisher": {
        "keywords": [
            "write", "publish", "format", "document",
            "blog", "post", "article", "markdown"
        ]
    },
    "autopublisher": {
        "keywords": [
            "auto-generate", "auto publish", "automate publishing"
        ]
    },
    "data_cleaner": {
        "keywords": [
            "clean data", "normalize", "standardize",
            "preprocess", "sanitize", "fix data"
        ]
    },
    "version_manager": {
        "keywords": [
            "version", "commit", "history", "changelog",
            "release", "tag", "semantic versioning"
        ]
    },
    "rube_adapter": {
        "keywords": [
            "convert", "transform", "adapt", "bridge",
            "interface", "mapping", "serialize"
        ]
    },
    "llm_connector": {
        "keywords": [
            "llm", "openai", "claude", "gemini", "qwen", "grok",
            "external model", "ai model", "chatgpt", "anthropic",
            "llama", "provider", "api call", "remote model"
        ]
    },
    "finance_handler": {
        "keywords": [
            "invoice", "bill", "payment", "finance", "money", "cost", "budget"
        ]
    },
    "debug_expert": {
        "keywords": [
            "error", "bug", "debug", "fix", "issue", "problem", "crash"
        ]
    },
    "support_agent": {
        "keywords": [
            "help", "support", "assist", "question", "guidance"
        ]
    },
    "calendar_flow": {
        "keywords": [
            "meeting", "schedule", "calendar", "appointment", "time"
        ]
    },
    "priority_handler": {
        "keywords": [
            "urgent", "emergency", "critical", "asap", "priority"
        ]
    }
}

# Intent patterns
INTENT_PATTERNS = {
    "invoice": ["invoice", "bill", "payment", "process"],
    "error": ["error", "bug", "issue", "problem", "fail"],
    "meeting": ["meeting", "schedule", "calendar", "appointment"],
    "support": ["help", "support", "assist", "question"],
    "urgent": ["urgent", "emergency", "critical", "asap"]
}


# Precompute word vectors for simple cosine similarity (no external API)
def text_to_vector(text: str) -> Dict[str, int]:
    words = re.findall(r"\w+", text.lower())
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    return freq


def cosine_sim(vec1: Dict[str, int], vec2: Dict[str, int]) -> float:
    common = set(vec1.keys()) & set(vec2.keys())
    num = sum(vec1[w] * vec2[w] for w in common)
    den = math.sqrt(sum(v*v for v in vec1.values())) * math.sqrt(sum(v*v for v in vec2.values()))
    return num / den if den != 0 else 0.0


class SemanticRouterExpert:
    """
    Intelligent router that:
    - Performs keyword matching
    - Uses local similarity scoring
    - Only returns valid experts
    - Supports fallback
    """

    def run(self, prompt: str) -> dict:
        prompt_vec = text_to_vector(prompt)
        expert_scores = {}
        matched_keywords = []

        # Determine intent
        intent = "unknown"
        for intent_type, patterns in INTENT_PATTERNS.items():
            for pattern in patterns:
                if pattern in prompt.lower():
                    intent = intent_type
                    break
            if intent != "unknown":
                break

        # Score against each expert
        for expert, config in EXPERTS.items():
            keywords = config["keywords"]
            kw_text = " ".join(keywords)
            kw_vec = text_to_vector(kw_text)
            score = cosine_sim(prompt_vec, kw_vec)

            # Track matches
            for kw in keywords:
                if kw in prompt.lower():
                    matched_keywords.append((expert, kw))

            expert_scores[expert] = score

        # Optional embeddings hint
        embeddings_boost = {}
        if EMBEDDINGS_AVAILABLE:
            try:
                prompt_embedding = embed_text(prompt)
                for expert, config in EXPERTS.items():
                    namespace = config.get("embedding_namespace")
                    if namespace:
                        results = search(prompt_embedding, namespace=namespace, top_k=1)
                        if results and results[0]["score"] > 0.75:  # Configurable threshold
                            boost = results[0]["score"] * 0.1  # Subtle boost
                            expert_scores[expert] += boost
                            embeddings_boost[expert] = round(boost, 4)
                            logging.debug(f"Embeddings boost for {expert}: +{boost}")
            except Exception as e:
                logging.debug(f"Embeddings hint failed: {e}")

        # Find best expert
        best_score = 0
        best_expert = "general_handler"
        for expert, score in expert_scores.items():
            if score > best_score:
                best_score = score
                best_expert = expert

        # Keyword match override
        if matched_keywords:
            # highest-frequency expert wins
            counts = {}
            for (exp, kw) in matched_keywords:
                counts[exp] = counts.get(exp, 0) + 1
            best_expert = max(counts, key=counts.get)

        # Threshold rule
        if best_score < 0.05 and not matched_keywords:
            best_expert = "general_handler"

        return {
            "input": prompt,
            "intent": intent,
            "route": best_expert,
            "score": round(best_score, 4),
            "keyword_hits": matched_keywords,
            "embeddings_boost": embeddings_boost
        }
