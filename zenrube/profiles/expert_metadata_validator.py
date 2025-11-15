"""
Expert Metadata Validator for Zenrube Team Council
Validates EXPERT_METADATA presence, structure, and version consistency

Author: vladinc@gmail.com
"""

import os
import json
import hashlib
import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class ExpertMetadataValidator:
    """
    Validates EXPERT_METADATA in all expert files.
    
    Responsibilities:
    1. Verify presence of EXPERT_METADATA in every expert file
    2. Verify valid fields + types
    3. Compute file hash
    4. Compare hash to stored hash in expert_metadata_state.json
    5. If hash changed AND version unchanged → error or auto-increment
    6. If metadata missing → error or auto-insert
    7. Update expert_metadata_state.json with new hashes + versions
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the validator."""
        self.experts_dir = "zenrube/experts"
        self.state_file = "zenrube/config/expert_metadata_state.json"
        
        # Default configuration
        self.config = {
            "strict_metadata_validation": True,
            "auto_fix_missing_metadata": True,
            "auto_fix_version_mismatch": False,
            **config
        } if config else {
            "strict_metadata_validation": True,
            "auto_fix_missing_metadata": True,
            "auto_fix_version_mismatch": False
        }
        
        self.state_data = {}
        self.validation_errors = []
        self.validation_warnings = []
        
        logger.info("ExpertMetadataValidator initialized")
    
    def validate_all(self) -> Dict[str, Any]:
        """
        Validate all expert files in the experts directory.
        
        Returns:
            Dict[str, Any]: Validation results
        """
        logger.info("Starting expert metadata validation for all files")
        
        try:
            # Load current state
            self._load_state()
            
            # Scan experts directory
            expert_files = self._scan_expert_files()
            logger.info(f"Found {len(expert_files)} expert files to validate")
            
            results = {
                "total_files": len(expert_files),
                "validated_files": 0,
                "errors": [],
                "warnings": [],
                "auto_fixes": [],
                "skipped_files": [],
                "validation_timestamp": self._get_timestamp()
            }
            
            # Validate each file
            for file_path in expert_files:
                try:
                    result = self._validate_single_file(file_path)
                    if result["status"] == "valid":
                        results["validated_files"] += 1
                    elif result["status"] == "error":
                        results["errors"].append(result)
                    elif result["status"] == "warning":
                        results["warnings"].append(result)
                    elif result["status"] == "auto_fixed":
                        results["auto_fixes"].append(result)
                    elif result["status"] == "skipped":
                        results["skipped_files"].append(result)
                        
                except Exception as e:
                    error_result = {
                        "file": file_path,
                        "status": "error",
                        "error": str(e),
                        "type": "validation_exception"
                    }
                    results["errors"].append(error_result)
            
            # Save updated state
            self._save_state()
            
            # Summary
            logger.info(f"Validation completed: {results['validated_files']} valid, "
                       f"{len(results['errors'])} errors, {len(results['warnings'])} warnings, "
                       f"{len(results['auto_fixes'])} auto-fixed")
            
            return results
            
        except Exception as e:
            logger.error(f"Expert metadata validation failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "validation_timestamp": self._get_timestamp()
            }
    
    def _scan_expert_files(self) -> List[str]:
        """Scan the experts directory for Python files."""
        expert_files = []
        
        if not os.path.exists(self.experts_dir):
            logger.warning(f"Experts directory not found: {self.experts_dir}")
            return expert_files
        
        for file_name in os.listdir(self.experts_dir):
            if file_name.endswith(".py") and not file_name.startswith("__"):
                expert_files.append(os.path.join(self.experts_dir, file_name))
        
        return sorted(expert_files)
    
    def _validate_single_file(self, file_path: str) -> Dict[str, Any]:
        """Validate a single expert file."""
        logger.debug(f"Validating file: {file_path}")
        
        file_name = os.path.basename(file_path)
        file_key = file_name.replace(".py", "")
        
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Compute file hash
            file_hash = hashlib.md5(content.encode()).hexdigest()
            
            # Extract metadata
            metadata_result = self._extract_expert_metadata(content, file_path)
            
            if metadata_result["status"] == "missing":
                return self._handle_missing_metadata(file_path, file_hash, file_key)
            
            elif metadata_result["status"] == "malformed":
                return self._handle_malformed_metadata(file_path, file_hash, file_key, metadata_result)
            
            elif metadata_result["status"] == "valid":
                return self._validate_existing_metadata(file_path, file_hash, file_key, metadata_result["metadata"])
            
            else:
                return {
                    "file": file_path,
                    "status": "error",
                    "error": f"Unknown metadata extraction status: {metadata_result['status']}"
                }
                
        except Exception as e:
            return {
                "file": file_path,
                "status": "error",
                "error": str(e),
                "type": "file_processing_error"
            }
    
    def _extract_expert_metadata(self, content: str, file_path: str) -> Dict[str, Any]:
        """Extract EXPERT_METADATA from file content."""
        # Look for EXPERT_METADATA declaration
        metadata_pattern = r'EXPERT_METADATA\s*=\s*(\{.*?\})'
        
        # Handle multi-line dictionary with proper indentation
        metadata_pattern_multi = r'EXPERT_METADATA\s*=\s*(\{.*?\n\s*\})'
        
        match = re.search(metadata_pattern, content, re.DOTALL)
        
        if not match:
            # Try alternative pattern
            match = re.search(metadata_pattern_multi, content, re.DOTALL)
        
        if not match:
            return {
                "status": "missing",
                "error": "EXPERT_METADATA not found in file"
            }
        
        try:
            # Parse the metadata dictionary
            metadata_str = match.group(1)
            
            # Use eval to parse the dictionary (safe in controlled environment)
            metadata = eval(metadata_str)
            
            # Validate structure
            validation_result = self._validate_metadata_structure(metadata)
            
            if not validation_result["valid"]:
                return {
                    "status": "malformed",
                    "error": f"Invalid metadata structure: {validation_result['errors']}",
                    "raw_metadata": metadata
                }
            
            return {
                "status": "valid",
                "metadata": metadata,
                "line_number": content[:match.start()].count('\n') + 1
            }
            
        except Exception as e:
            return {
                "status": "malformed",
                "error": f"Failed to parse EXPERT_METADATA: {str(e)}",
                "raw_metadata_str": match.group(1) if match else None
            }
    
    def _validate_metadata_structure(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the structure of EXPERT_METADATA."""
        errors = []
        
        # Required fields
        required_fields = ["name", "version", "description", "author"]
        for field in required_fields:
            if field not in metadata:
                errors.append(f"Missing required field: {field}")
        
        # Field types
        if "name" in metadata and not isinstance(metadata["name"], str):
            errors.append("Field 'name' must be a string")
        
        if "version" in metadata:
            if not isinstance(metadata["version"], int):
                errors.append("Field 'version' must be an integer")
            elif metadata["version"] < 1:
                errors.append("Field 'version' must be >= 1")
        
        if "description" in metadata and not isinstance(metadata["description"], str):
            errors.append("Field 'description' must be a string")
        
        if "author" in metadata and not isinstance(metadata["author"], str):
            errors.append("Field 'author' must be a string")
        
        # Optional validation: description length
        if "description" in metadata and len(metadata["description"]) > 200:
            errors.append("Field 'description' should be under 200 characters")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    def _handle_missing_metadata(self, file_path: str, file_hash: str, file_key: str) -> Dict[str, Any]:
        """Handle missing EXPERT_METADATA."""
        if self.config["auto_fix_missing_metadata"] and not self.config["strict_metadata_validation"]:
            # Auto-insert metadata
            return self._auto_insert_metadata(file_path, file_hash, file_key)
        else:
            # Report error
            return {
                "file": file_path,
                "status": "error",
                "error": "EXPERT_METADATA is missing",
                "type": "missing_metadata",
                "fix_suggested": self.config["auto_fix_missing_metadata"]
            }
    
    def _auto_insert_metadata(self, file_path: str, file_hash: str, file_key: str) -> Dict[str, Any]:
        """Auto-insert missing EXPERT_METADATA."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create metadata
            metadata = {
                "name": file_key,
                "version": 1,
                "description": f"Auto-inserted metadata for {file_key} expert.",
                "author": "vladinc@gmail.com"
            }
            
            # Insert after imports or at the beginning
            metadata_str = f"EXPERT_METADATA = {json.dumps(metadata, indent=4)}\n\n"
            
            # Find insertion point (after imports but before classes)
            lines = content.split('\n')
            insert_line = 0
            
            for i, line in enumerate(lines):
                line_stripped = line.strip()
                if (line_stripped.startswith('import ') or 
                    line_stripped.startswith('from ') or
                    line_stripped.startswith('#') or
                    line_stripped.startswith('"""') or
                    line_stripped == ''):
                    insert_line = i + 1
                else:
                    break
            
            # Insert metadata
            lines.insert(insert_line, metadata_str.rstrip())
            
            # Write back
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            
            # Update state
            self.state_data[file_key] = {
                "file_path": file_path,
                "file_hash": file_hash,
                "version": metadata["version"],
                "last_validated": self._get_timestamp(),
                "metadata": metadata
            }
            
            logger.info(f"Auto-inserted EXPERT_METADATA into {file_path}")
            
            return {
                "file": file_path,
                "status": "auto_fixed",
                "action": "metadata_inserted",
                "metadata": metadata
            }
            
        except Exception as e:
            return {
                "file": file_path,
                "status": "error",
                "error": f"Failed to auto-insert metadata: {str(e)}"
            }
    
    def _handle_malformed_metadata(self, file_path: str, file_hash: str, file_key: str, 
                                 metadata_result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle malformed EXPERT_METADATA."""
        return {
            "file": file_path,
            "status": "error",
            "error": metadata_result["error"],
            "type": "malformed_metadata",
            "raw_metadata": metadata_result.get("raw_metadata")
        }
    
    def _validate_existing_metadata(self, file_path: str, file_hash: str, file_key: str, 
                                  metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate existing EXPERT_METADATA and check for version changes."""
        
        # Check if file content changed
        previous_state = self.state_data.get(file_key, {})
        previous_hash = previous_state.get("file_hash")
        previous_version = previous_state.get("version")
        
        version = metadata["version"]
        name = metadata["name"]
        
        # Validate that name matches filename
        if name != file_key:
            return {
                "file": file_path,
                "status": "warning",
                "error": f"Metadata name '{name}' doesn't match filename '{file_key}'",
                "type": "name_mismatch"
            }
        
        # Check for version mismatch (content changed but version didn't)
        if previous_hash and previous_hash != file_hash and previous_version == version:
            if self.config["strict_metadata_validation"]:
                return {
                    "file": file_path,
                    "status": "error",
                    "error": f"File content changed (hash: {file_hash[:8]}...) but version unchanged ({version})",
                    "type": "version_mismatch",
                    "action_required": "increment_version"
                }
            elif self.config["auto_fix_version_mismatch"]:
                return self._auto_increment_version(file_path, file_hash, file_key, metadata)
            else:
                return {
                    "file": file_path,
                    "status": "warning",
                    "error": f"File content changed but version unchanged",
                    "type": "version_mismatch"
                }
        
        # Update state
        self.state_data[file_key] = {
            "file_path": file_path,
            "file_hash": file_hash,
            "version": version,
            "last_validated": self._get_timestamp(),
            "metadata": metadata
        }
        
        return {
            "file": file_path,
            "status": "valid",
            "version": version,
            "name": name,
            "file_hash": file_hash[:8] + "...",
            "content_changed": previous_hash != file_hash if previous_hash else True
        }
    
    def _auto_increment_version(self, file_path: str, file_hash: str, file_key: str, 
                              metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Auto-increment version when content changes."""
        try:
            # Increment version
            new_version = metadata["version"] + 1
            metadata["version"] = new_version
            
            # Update in file
            self._update_metadata_in_file(file_path, metadata)
            
            # Update state
            self.state_data[file_key] = {
                "file_path": file_path,
                "file_hash": file_hash,
                "version": new_version,
                "last_validated": self._get_timestamp(),
                "metadata": metadata
            }
            
            logger.info(f"Auto-incremented version for {file_path} to {new_version}")
            
            return {
                "file": file_path,
                "status": "auto_fixed",
                "action": "version_incremented",
                "old_version": new_version - 1,
                "new_version": new_version
            }
            
        except Exception as e:
            return {
                "file": file_path,
                "status": "error",
                "error": f"Failed to auto-increment version: {str(e)}"
            }
    
    def _update_metadata_in_file(self, file_path: str, metadata: Dict[str, Any]) -> None:
        """Update EXPERT_METADATA in file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace metadata
        metadata_str = json.dumps(metadata, indent=4)
        pattern = r'EXPERT_METADATA\s*=\s*(\{.*?\})'
        
        def replace_metadata(match):
            return f'EXPERT_METADATA = {metadata_str}'
        
        new_content = re.sub(pattern, replace_metadata, content, flags=re.DOTALL)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
    
    def _load_state(self) -> None:
        """Load validation state from file."""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    self.state_data = json.load(f)
                logger.debug(f"Loaded state data: {len(self.state_data)} files")
            else:
                self.state_data = {}
                logger.debug("No existing state file, starting fresh")
        except Exception as e:
            logger.warning(f"Failed to load state file: {e}")
            self.state_data = {}
    
    def _save_state(self) -> None:
        """Save validation state to file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.state_data, f, indent=2, ensure_ascii=False)
            logger.debug(f"Saved state data: {len(self.state_data)} files")
        except Exception as e:
            logger.error(f"Failed to save state file: {e}")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation results."""
        return {
            "config": self.config,
            "state_files_count": len(self.state_data),
            "last_validation": self.state_data.get("__last_validation_timestamp"),
            "validation_errors": getattr(self, 'validation_errors', []),
            "validation_warnings": getattr(self, 'validation_warnings', [])
        }
    
    def force_revalidation(self, file_path: str) -> Dict[str, Any]:
        """Force revalidation of a specific file."""
        logger.info(f"Force revalidating file: {file_path}")
        
        # Remove from state to force fresh validation
        file_key = os.path.basename(file_path).replace(".py", "")
        if file_key in self.state_data:
            del self.state_data[file_key]
        
        return self._validate_single_file(file_path)
    
    def get_expert_info(self, expert_name: str) -> Optional[Dict[str, Any]]:
        """Get metadata info for a specific expert."""
        file_path = os.path.join(self.experts_dir, f"{expert_name}.py")
        
        if not os.path.exists(file_path):
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            result = self._extract_expert_metadata(content, file_path)
            
            if result["status"] == "valid":
                return {
                    "name": result["metadata"]["name"],
                    "version": result["metadata"]["version"],
                    "description": result["metadata"]["description"],
                    "author": result["metadata"]["author"],
                    "file_path": file_path,
                    "line_number": result.get("line_number")
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get expert info for {expert_name}: {e}")
            return None