import re
import logging
from typing import List, Dict

EXPERT_METADATA = {
    "name": "summarizer",
    "version": "1.0",
    "description": "Summarizes long text into concise overviews using simple heuristics.",
    "author": "vladinc@gmail.com"
}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SummarizerExpert:
    """
    Expert class for summarizing long text inputs into concise, meaningful summaries.
    
    Uses simple heuristic-based approach to select the most informative sentences
    from the input text, preferring sentences that contain important keywords.
    """
    
    def __init__(self):
        # Keywords that indicate important/summary sentences
        self.important_keywords = [
            "important", "key", "summary", "result", "conclusion", 
            "final", "ultimately", "therefore", "thus", "hence",
            "overall", "in summary", "in conclusion", "main point"
        ]
    
    def run(self, input_data: str, max_sentences: int = 3) -> str:
        """
        Summarize the input text by selecting the most informative sentences.
        
        Args:
            input_data (str): Plain text input to summarize (paragraph or multiple paragraphs).
            max_sentences (int): Maximum number of sentences to include in summary (default: 3).
        
        Returns:
            str: A condensed summary of the input text, or empty string for invalid input.
        """
        # Validate input
        if not isinstance(input_data, str):
            logger.warning("Non-string input received, converting to string.")
            input_data = str(input_data)
        
        if not input_data or not input_data.strip():
            logger.info("Empty or invalid input received, returning empty string.")
            return ""
        
        logger.info(f"Summarizing text of {len(input_data)} characters (max_sentences={max_sentences})")
        
        # Split text into sentences
        sentences = self._split_into_sentences(input_data)
        
        if not sentences:
            logger.info("No sentences found in input, returning empty string.")
            return ""
        
        # If input is shorter than max_sentences, return it unchanged
        if len(sentences) <= max_sentences:
            logger.info(f"Input has {len(sentences)} sentences, returning unchanged (≤ {max_sentences})")
            return input_data.strip()
        
        # Select most informative sentences
        selected_sentences = self._select_informative_sentences(sentences, max_sentences)
        
        # Reassemble summary
        summary = " ".join(selected_sentences).strip()
        
        logger.info(f"Generated summary with {len(selected_sentences)} sentences from {len(sentences)} original")
        return summary
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using simple punctuation-based rules.
        
        Args:
            text (str): Input text to split.
        
        Returns:
            List[str]: List of sentences found in the text.
        """
        # Simple regex to split on sentence-ending punctuation
        # Handles ., !, ? followed by space or end of string
        sentence_pattern = r'[.!?]+\s+|\.|\!|\?$'
        sentences = re.split(sentence_pattern, text)
        
        # Clean up sentences: remove extra whitespace and empty strings
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:  # Only add non-empty sentences
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _select_informative_sentences(self, sentences: List[str], max_sentences: int) -> List[str]:
        """
        Select the most informative sentences using heuristics.
        
        Prioritizes sentences containing important keywords, otherwise selects
        the first N sentences up to max_sentences.
        
        Args:
            sentences (List[str]): List of sentences to select from.
            max_sentences (int): Maximum number of sentences to select.
        
        Returns:
            List[str]: Selected sentences in original order.
        """
        if len(sentences) <= max_sentences:
            return sentences
        
        # First, try to find sentences with important keywords
        keyword_sentences = []
        regular_sentences = []
        
        for sentence in sentences:
            if self._contains_important_keywords(sentence):
                keyword_sentences.append(sentence)
            else:
                regular_sentences.append(sentence)
        
        selected = []
        
        # Prioritize sentences with important keywords
        for sentence in keyword_sentences:
            if len(selected) < max_sentences:
                selected.append(sentence)
        
        # If we still need more sentences, add from the beginning of regular sentences
        if len(selected) < max_sentences:
            remaining_needed = max_sentences - len(selected)
            # Take from the beginning to maintain context flow
            for sentence in regular_sentences[:remaining_needed]:
                if sentence not in selected:
                    selected.append(sentence)
        
        # If we still don't have enough (due to duplicates), fill with remaining sentences
        if len(selected) < max_sentences:
            remaining_needed = max_sentences - len(selected)
            for sentence in regular_sentences:
                if sentence not in selected and remaining_needed > 0:
                    selected.append(sentence)
                    remaining_needed -= 1
        
        logger.info(f"Selected {len(selected)} sentences: {len(keyword_sentences)} with keywords, {len(selected) - len(keyword_sentences) if len(selected) > len(keyword_sentences) else 0} from regular")
        return selected[:max_sentences]  # Ensure we don't exceed max_sentences
    
    def _contains_important_keywords(self, sentence: str) -> bool:
        """
        Check if a sentence contains any of the important keywords.
        
        Args:
            sentence (str): Sentence to check.
        
        Returns:
            bool: True if sentence contains important keywords, False otherwise.
        """
        sentence_lower = sentence.lower()
        
        for keyword in self.important_keywords:
            # Use word boundaries to match whole words, not partial matches
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, sentence_lower):
                logger.debug(f"Found keyword '{keyword}' in sentence: {sentence[:50]}...")
                return True
        
        return False