#!/usr/bin/env python3
"""
Clean text-based files before committing to GitHub.
Removes emojis, personal info, and adds professional headers.
Applies to all text files (not just .py scripts) across the repository.
"""

import re
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Personal info or sensitive data to redact
PERSONAL_INFO = {
    '[REDACTED_EMAIL]': '[REDACTED_EMAIL]',
    '[REDACTED_NAME]': '[REDACTED_NAME]',
    '[REDACTED_NAME]': '[REDACTED_NAME]',
    '[REDACTED_NAME]': '[REDACTED_NAME]',
    '[REDACTED_USER]': '[REDACTED_USER]',
    '[REDACTED_USER]': '[REDACTED_USER]',
    '/path/to/user': '/path/to/user',
    '/path/to/user': '/path/to/user',
    '[REDACTED_AFFILIATION]': '[REDACTED_AFFILIATION]',
    '[REDACTED_AFFILIATION]': '[REDACTED_AFFILIATION]',
    '[REDACTED_AFFILIATION]': '[REDACTED_AFFILIATION]',
    '[REDACTED_AFFILIATION]': '[REDACTED_AFFILIATION]',
    '[REDACTED_CREDENTIAL]': '[REDACTED_CREDENTIAL]',
    '[REDACTED_CREDENTIAL]': '[REDACTED_CREDENTIAL]',
    '[REDACTED_CREDENTIAL]': '[REDACTED_CREDENTIAL]',
}

# Emoji pattern to remove
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

# Text file extensions to include (beyond .py)
TEXT_EXTS = {'.py', '.md', '.txt', '.json', '.yml', '.yaml', '.ini', '.cfg', '.log', '.csv'}

def is_text_file(file_path: Path) -> bool:
    """Heuristic check for text-based files."""
    if file_path.suffix.lower() in TEXT_EXTS:
        return True
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(1024)
        return all(32 <= b < 127 or b in (9, 10, 13) for b in chunk)
    except Exception:
        return False

def clean_file(file_path: Path) -> tuple[bool, str]:
    """
    Clean text-based file.
    Returns (changed, message)
    """
    try:
        if not is_text_file(file_path):
            return False, "Skipped (binary)"

        content = file_path.read_text(encoding='utf-8', errors='ignore')
        original_content = content

        # Remove emojis
        content = EMOJI_PATTERN.sub('', content)

        # Replace personal info
        for old, new in PERSONAL_INFO.items():
            content = content.replace(old, new)

        # Remove excessively long comment lines
        lines = content.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#') and len(stripped) > 100:
                cleaned_lines.append(line[:100] + '...')
            else:
                cleaned_lines.append(line)
        content = '\n'.join(cleaned_lines)

        # Save if changed
        if content != original_content:
            file_path.write_text(content, encoding='utf-8')
            return True, "Cleaned"
        else:
            return False, "No changes needed"

    except Exception as e:
        return False, f"Error: {e}"

def main():
    """Clean all text files in the repository."""
    print("=" * 90)
    print(" ARCTIC ZERO CURTAIN PIPELINE — FULL REPOSITORY SANITIZATION ")
    print("=" * 90)

    # All text-based files (not limited to .py)
    all_files = [
        f for f in PROJECT_ROOT.rglob('*')
        if f.is_file() and not any(x in f.parts for x in ['.git', '__pycache__', 'venv', 'env', 'isce3'])
    ]

    print(f"\nFound {len(all_files)} files to inspect.\n")

    cleaned_count = 0
    for file in all_files:
        rel_path = file.relative_to(PROJECT_ROOT)
        changed, message = clean_file(file)
        print(f"{rel_path}: {message}")
        if changed:
            cleaned_count += 1

    print("\n" + "=" * 90)
    print(f"SUMMARY: Cleaned {cleaned_count} files across repository")
    print("=" * 90)

if __name__ == "__main__":
    main()
