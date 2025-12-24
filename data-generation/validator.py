import re
import json
from typing import Dict, List, Tuple, Any, Optional

class OutputValidator:
    def __init__(self):
        # A2. Forbidden patterns (numbers, units, stats)
        # We allow some generic words, but ban specific units/stats terms
        # Relaxed: Removed generic digit check (\d) because f0, 46-55 etc contain digits.
        # Instead we ban specific units and statistical terms.
        self.forbidden_patterns = [
            r"(?i)\b(Hz|kHz|ms|sec|seconds)\b", 
        ]
        
        # A1. Format Regex
        # <THINK>... </THINK> <RESPONSE>... </RESPONSE>
        # Dotall to capture newlines inside tags
        self.format_regex = re.compile(
            r"<THINK>(.*?)</THINK>\s*<RESPONSE>(.*?)</RESPONSE>", 
            re.DOTALL | re.IGNORECASE
        )
        
        # Response Template Regex
        # "Based on features such as trait1=val1, trait2=val2, the speaker is most consistent with: LABEL."
        # Allowing some flexibility in whitespace and optional trailing period
        self.response_template_regex = re.compile(
            r"Based on features such as\s+(.+?),\s+the speaker is most consistent with:\s+(.+?)[\.\s]*$",
            re.IGNORECASE | re.MULTILINE
        )

    def parse_blocks(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        match = self.format_regex.search(text)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        return None, None

    def _check_forbidden(self, text: str) -> List[str]:
        violations = []
        for pat in self.forbidden_patterns:
            matches = re.findall(pat, text)
            if matches:
                violations.append(f"Forbidden pattern found: {matches[0]}")
        return violations

    def _extract_kv_pairs(self, text: str) -> List[Tuple[str, str]]:
        # Naive: key=value
        # Better: \b(\w+)=([\w\|]+)\b
        # Assuming keys are simple words/underscores, values are simple words/pipes
        found = re.findall(r"(\w+)=([\w\|]+)", text)
        return found

    def hard_validate(self, 
                      text: str, 
                      latent_json: Dict[str, Any], 
                      candidates: List[str]) -> Tuple[bool, List[str]]:
        """
        Returns (is_valid, list_of_violations)
        """
        violations = []
        
        # 1. Format Check
        think, response = self.parse_blocks(text)
        if not think or not response:
            return False, ["Format Error: Missing <THINK> or <RESPONSE> tags"]
            
        # 2. Forbidden Content (Numbers/Stats) in THINK
        violations.extend(self._check_forbidden(think))
        
        # 3. Trait Allowlist Check (in THINK and RESPONSE)
        # Flatten latent keys/values for easy check
        valid_pairs = set()
        for cat, content in latent_json.items():
            if isinstance(content, dict):
                for k, v in content.items():
                    valid_pairs.add((k, str(v)))
            # Handle non-nested if any (version string etc)
        
        # Check pairs in THINK
        think_pairs = self._extract_kv_pairs(think)
        for k, v in think_pairs:
            if (k, v) not in valid_pairs:
                violations.append(f"Invalid trait in THINK: {k}={v}")

        # 4. Response Template & Label Check
        # Regex check
        resp_match = self.response_template_regex.search(response) 
        
        label = ""
        traits_str = ""
        
        if resp_match:
            traits_str, label = resp_match.groups()
        else:
            # Fallback Parsing Strategy
            if "consistent with:" in response:
                try:
                    # Extract label
                    label_part = response.split("consistent with:")[-1].strip()
                    label = label_part.split(".")[0].strip() # Take everything up to the first period
                    
                    # If regex failed, we can't easily validate the "Based on features..." structure
                    # But we'll let it slide if we can find valid traits in the whole response
                    # However, strictly enforcing the template is better for uniformity.
                    violations.append("Response Template Error: Structure mismatch. Expected: 'Based on features such as ..., the speaker is most consistent with: LABEL.'")
                except:
                    violations.append("Response Template Error: Could not parse response structure.")
                    return False, violations
            else:
                violations.append("Response Template Error: Must match 'Based on features such as ..., the speaker is most consistent with: LABEL.'")
                return False, violations

        # Check traits in Response (only if regex matched)
        if resp_match:
            resp_kv = self._extract_kv_pairs(traits_str)
            
            if len(resp_kv) < 2:
                 violations.append(f"Insufficient traits in RESPONSE: Found {len(resp_kv)}, expected at least 2.")
            
            for k, v in resp_kv:
                if (k, v) not in valid_pairs:
                     violations.append(f"Invalid trait in RESPONSE: {k}={v}")
                     
        # Check Label
        # Aggressive cleanup: remove non-alphanumeric except spaces and hyphens (for ranges like 46-55)
        clean_label = re.sub(r"[^a-zA-Z0-9\s\-/]", "", label).strip()
        
        # Robust Check: 
        candidates_lower = [c.lower() for c in candidates]
        label_lower = clean_label.lower()
        
        matched = False
        if label_lower in candidates_lower:
            matched = True
        else:
            # Fuzzy Check
            for cand in candidates_lower:
                if cand == label_lower:
                    matched = True
                    break
                if len(cand) > 3 and cand in label_lower: 
                    matched = True
                    break
                if len(label_lower) > 3 and label_lower in cand:
                    matched = True
                    break
        
        if not matched:
             violations.append(f"Invalid Label: '{clean_label}' (raw: '{label}') not in candidates: {candidates}")

        return (len(violations) == 0), violations
