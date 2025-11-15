"""
Dynamic Classification Engine for Zenrube Team Council
Two-step classification: keyword classifier + semantic router fallback

Author: vladinc@gmail.com
"""

import logging
from typing import Dict, Any, Optional
import re

logger = logging.getLogger(__name__)


class ClassificationEngine:
    """
    Two-step classification engine for task categorization.
    
    Step 1: Fast keyword-based classification
    Step 2: Semantic router fallback for complex cases
    """
    
    def __init__(self):
        """Initialize the classification engine."""
        self._setup_keyword_patterns()
        logger.info("ClassificationEngine initialized")
    
    def _setup_keyword_patterns(self):
        """Setup keyword patterns for fast classification."""
        self.keyword_domains = {
            "cybersec": {
                "keywords": ["security", "vulnerability", "hack", "attack", "breach", "cyber", "malware", "threat", "risk", "encryption", "auth", "firewall", "antivirus", "spyware", "phishing", "ddos", "ssl", "tls", "penetration", "exploit"],
                "weight": 3
            },
            "coding": {
                "keywords": ["code", "programming", "develop", "debug", "implement", "algorithm", "function", "class", "method", "api", "framework", "library", "compile", "syntax", "bug", "refactor", "optimize", "test"],
                "weight": 3
            },
            "creative": {
                "keywords": ["design", "creative", "artistic", "visual", "ui", "ux", "prototype", "mockup", "concept", "brainstorm", "innovative", "story", "narrative", "brand", "campaign", "content", "marketing"],
                "weight": 2
            },
            "feelprint": {
                "keywords": ["emotion", "feeling", "sentiment", "mood", "experience", "user", "human", "psychology", "behavior", "personality", "empathy", "touch", "warm", "cold", "friendly", "hostile"],
                "weight": 2
            },
            "hardware": {
                "keywords": ["hardware", "device", "chip", "circuit", "board", "sensor", "processor", "memory", "storage", "network", "infrastructure", "server", "cloud", "iot", "embedded", "firmware"],
                "weight": 2
            },
            "data": {
                "keywords": ["data", "database", "analytics", "statistics", "machine learning", "model", "algorithm", "predict", "insight", "pattern", "analysis", "csv", "json", "sql", "big data"],
                "weight": 3
            },
            "business": {
                "keywords": ["business", "strategy", "market", "revenue", "customer", "product", "sales", "growth", "roi", "kpi", "metric", "goal", "objective", "plan", "budget"],
                "weight": 1
            }
        }
        
        self.domain_signals = {
            "cybersec": ["technical", "security-focused", "risk-aware"],
            "coding": ["technical", "implementation-focused", "code-oriented"],
            "creative": ["design-focused", "visual", "innovative"],
            "feelprint": ["human-centered", "emotional", "experience-focused"],
            "hardware": ["technical", "physical", "device-oriented"],
            "data": ["analytical", "quantitative", "insight-driven"],
            "business": ["strategic", "commercial", "goal-oriented"]
        }
    
    def classify_task(self, task: str) -> Dict[str, Any]:
        """
        Classify a task using two-step approach.
        
        Args:
            task (str): Task description to classify
            
        Returns:
            Dict[str, Any]: Classification results with primary, secondary, confidence, signals
        """
        task_lower = task.lower()
        
        # Step 1: Fast keyword classification
        keyword_scores = self._score_keyword_matches(task_lower)
        
        if keyword_scores:
            primary_domain = max(keyword_scores.keys(), key=lambda k: keyword_scores[k]["score"])
            primary_score = keyword_scores[primary_domain]["score"]
            
            # Find secondary domain
            secondary_domain = None
            secondary_score = 0
            
            for domain, score_info in keyword_scores.items():
                if domain != primary_domain and score_info["score"] > secondary_score:
                    secondary_domain = domain
                    secondary_score = score_info["score"]
            
            # Calculate confidence based on score strength
            confidence = min(0.9, primary_score / 10.0)  # Normalize to 0-0.9
            
            # Add semantic router fallback for complex cases
            if confidence < 0.5:
                return self._semantic_router_fallback(task, keyword_scores)
            
            signals = self._extract_signals(task_lower, primary_domain, secondary_domain)
            
            return {
                "primary": primary_domain,
                "secondary": secondary_domain,
                "confidence": confidence,
                "signals": signals,
                "method": "keyword_classification"
            }
        
        else:
            # Step 2: Semantic router fallback
            return self._semantic_router_fallback(task, {})
    
    def _score_keyword_matches(self, task_lower: str) -> Dict[str, Any]:
        """Score task against keyword patterns."""
        scores = {}
        
        for domain, config in self.keyword_domains.items():
            score = 0
            matched_keywords = []
            
            for keyword in config["keywords"]:
                if keyword in task_lower:
                    score += config["weight"]
                    matched_keywords.append(keyword)
            
            if score > 0:
                scores[domain] = {
                    "score": score,
                    "matched_keywords": matched_keywords,
                    "pattern_count": len(matched_keywords)
                }
        
        return scores
    
    def _semantic_router_fallback(self, task: str, keyword_scores: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback classification using semantic analysis."""
        # Simple semantic patterns for fallback
        semantic_patterns = {
            "technical": ["build", "create", "develop", "implement", "system", "architecture", "technical", "api"],
            "creative": ["design", "visual", "creative", "art", "style", "look", "feel", "brand"],
            "analytical": ["analyze", "data", "insight", "trend", "pattern", "statistics", "metric"],
            "security": ["secure", "safe", "protect", "vulnerable", "risk", "threat", "encryption"],
            "business": ["strategy", "business", "market", "customer", "revenue", "growth", "goal"]
        }
        
        task_lower = task.lower()
        semantic_scores = {}
        
        for category, patterns in semantic_patterns.items():
            score = sum(1 for pattern in patterns if pattern in task_lower)
            if score > 0:
                semantic_scores[category] = score
        
        # Map semantic categories to domains
        domain_mapping = {
            "technical": "coding",
            "creative": "creative", 
            "analytical": "data",
            "security": "cybersec",
            "business": "business"
        }
        
        # Combine with keyword scores if available
        combined_scores = keyword_scores.copy()
        
        for semantic_cat, score in semantic_scores.items():
            if semantic_cat in domain_mapping:
                domain = domain_mapping[semantic_cat]
                if domain in combined_scores:
                    combined_scores[domain]["score"] += score
                else:
                    combined_scores[domain] = {
                        "score": score,
                        "matched_keywords": [],
                        "pattern_count": 1
                    }
        
        if combined_scores:
            primary_domain = max(combined_scores.keys(), key=lambda k: combined_scores[k]["score"])
            confidence = min(0.7, combined_scores[primary_domain]["score"] / 8.0)
            
            return {
                "primary": primary_domain,
                "secondary": None,
                "confidence": confidence,
                "signals": {"method": "semantic_fallback", "fallback_reason": "low_keyword_confidence"},
                "method": "semantic_classification"
            }
        
        # Ultimate fallback
        return {
            "primary": "general",
            "secondary": None,
            "confidence": 0.3,
            "signals": {"method": "default_fallback", "fallback_reason": "no_classification_patterns"},
            "method": "default_classification"
        }
    
    def _extract_signals(self, task_lower: str, primary: str, secondary: str) -> Dict[str, Any]:
        """Extract signals from the classification."""
        signals = {
            "primary_domain_signals": self.domain_signals.get(primary, []),
            "secondary_domain_signals": self.domain_signals.get(secondary, []) if secondary else [],
            "task_characteristics": []
        }
        
        # Add task characteristics based on keywords found
        if any(word in task_lower for word in ["urgent", "quick", "fast", "asap"]):
            signals["task_characteristics"].append("time-sensitive")
        
        if any(word in task_lower for word in ["simple", "basic", "easy", "straightforward"]):
            signals["task_characteristics"].append("low-complexity")
        
        if any(word in task_lower for word in ["complex", "advanced", "sophisticated", "comprehensive"]):
            signals["task_characteristics"].append("high-complexity")
        
        if any(word in task_lower for word in ["collaborative", "team", "group", "multiple"]):
            signals["task_characteristics"].append("collaborative")
        
        return signals