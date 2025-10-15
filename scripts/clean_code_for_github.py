#!/usr/bin/env python3
"""
Clean Python files before committing to GitHub.
Removes emojis, personal info, and adds professional headers.
"""

import re
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Personal info to redact
PERSONAL_INFO = {
    'your.email@nasa.gov': 'your.email@nasa.gov',
    'Author Name': 'Author Name',
    'Dr. [Author]': 'Dr. [Author]',
    'username': 'username',
    '/Users/username': '/path/to/user',
}

# Emoji patterns to remove
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE
)

def clean_file(file_path: Path) -> tuple[bool, str]:
    """
    Clean a Python file.
    Returns (changed, message)
    """
    try:
        content = file_path.read_text(encoding='utf-8')
        original_content = content
        
        # Remove emojis
        content = EMOJI_PATTERN.sub('', content)
        
        # Replace personal info
        for old, new in PERSONAL_INFO.items():
            content = content.replace(old, new)
        
        # Remove excessive comments (lines starting with # that are > 100 chars)
        lines = content.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#') and len(stripped) > 100:
                # Shorten long comments
                cleaned_lines.append(line[:100] + '...')
            else:
                cleaned_lines.append(line)
        content = '\n'.join(cleaned_lines)
        
        # Check if changed
        if content != original_content:
            file_path.write_text(content, encoding='utf-8')
            return True, "Cleaned"
        else:
            return False, "No changes needed"
    
    except Exception as e:
        return False, f"Error: {e}"

def main():
    """Clean all Python files in the project."""
    print("=" * 70)
    print("CLEANING CODE FOR GITHUB")
    print("=" * 70)
    
    # Find all Python files
    py_files = list(PROJECT_ROOT.rglob("*.py"))
    
    # Exclude virtual environments and cache
    py_files = [f for f in py_files if not any(
        part in f.parts for part in ['venv', 'env', '__pycache__', '.git', 'isce3']
    )]
    
    print(f"\nFound {len(py_files)} Python files to check\n")
    
    cleaned_count = 0
    for py_file in py_files:
        rel_path = py_file.relative_to(PROJECT_ROOT)
        changed, message = clean_file(py_file)
        
        if changed:
            print(f" {rel_path}: {message}")
            cleaned_count += 1
        else:
            print(f"  {rel_path}: {message}")
    
    print(f"\n{'='*70}")
    print(f"SUMMARY: Cleaned {cleaned_count} files")
    print("=" * 70)

if __name__ == "__main__":
    main()
