"""Deterministic Validators for AURA Platform

Provides strict validation logic to supplement LLM evaluation:
- Regex matching
- Keyword presence/absence
- Numeric constraints
- Exact matches
"""

import re
from typing import Any, Dict, List, Optional, Union
import structlog

logger = structlog.get_logger()


def validate_regex_match(text: str, pattern: str) -> bool:
    """Validate if text matches a regex pattern"""
    try:
        return bool(re.search(pattern, text))
    except re.error as e:
        logger.error("Invalid regex pattern", pattern=pattern, error=str(e))
        return False


def validate_keyword_presence(text: str, keywords: List[str], case_sensitive: bool = False) -> bool:
    """Validate if ANY of the keywords are present"""
    if not case_sensitive:
        text = text.lower()
        keywords = [k.lower() for k in keywords]
        
    return any(k in text for k in keywords)


def validate_keyword_absence(text: str, keywords: List[str], case_sensitive: bool = False) -> bool:
    """Validate if NONE of the keywords are present (e.g., toxic words)"""
    if not case_sensitive:
        text = text.lower()
        keywords = [k.lower() for k in keywords]
        
    found = [k for k in keywords if k in text]
    if found:
        logger.debug("Forbidden keywords found", keywords=found)
        return False
    return True


def validate_numeric_range(
    value: Union[int, float, str], 
    min_val: Optional[float] = None, 
    max_val: Optional[float] = None
) -> bool:
    """Validate if a value is within a numeric range"""
    try:
        # Try to parse string numbers if needed
        if isinstance(value, str):
            value = float(value)
            
        if min_val is not None and value < min_val:
            return False
            
        if max_val is not None and value > max_val:
            return False
            
        return True
    except ValueError:
        return False


class DeterministicValidator:
    """Registry of deterministic validation methods"""
    
    @staticmethod
    def validate(
        input_text: str, 
        rule_type: str, 
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a deterministic validation rule.
        
        Returns:
            Dict with 'passed' (bool) and 'details' (str)
        """
        try:
            if rule_type == "regex_match":
                pattern = parameters.get("pattern", "")
                passed = validate_regex_match(input_text, pattern)
                return {
                    "passed": passed,
                    "details": f"Regex match '{pattern}': {'Found' if passed else 'Not Found'}"
                }
                
            elif rule_type == "regex_no_match":
                pattern = parameters.get("pattern", "")
                match = validate_regex_match(input_text, pattern)
                return {
                    "passed": not match,
                    "details": f"Forbidden regex '{pattern}': {'Found' if match else 'Not Found'}"
                }
                
            elif rule_type == "keyword_present":
                keywords = parameters.get("keywords", [])
                passed = validate_keyword_presence(input_text, keywords)
                return {
                    "passed": passed,
                    "details": f"Required keywords {keywords}: {'Found' if passed else 'Missing'}"
                }
                
            elif rule_type == "keyword_absent":
                keywords = parameters.get("keywords", [])
                passed = validate_keyword_absence(input_text, keywords)
                return {
                    "passed": passed,
                    "details": f"Forbidden keywords {keywords}: {'Absent' if passed else 'Found'}"
                }
                
            else:
                return {
                    "passed": False,
                    "details": f"Unknown rule type: {rule_type}"
                }
                
        except Exception as e:
            logger.error("Validation error", rule=rule_type, error=str(e))
            return {
                "passed": False,
                "details": f"Validation error: {str(e)}"
            }
