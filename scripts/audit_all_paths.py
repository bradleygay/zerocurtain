#!/usr/bin/env python3
"""
Comprehensive audit of ALL file paths in the repository.
Checks every file type for old path patterns.
"""

import re
from pathlib import Path
from typing import Dict, List, Set

class PathAuditor:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.issues = []
        
        # Define old patterns to search for
        self.old_patterns = {
            'imports': [
                r'from physics_detection\.',
                r'import physics_detection\.',
                r'from detection\.',
                r'import detection\.',
                r'from modeling\.',
                r'import modeling\.',
                r'from data_ingestion\.',
                r'import data_ingestion\.',
                r'from processing\.',
                r'import processing\.',
                r'from transformations\.',
                r'import transformations\.',
                r'from visualization\.',
                r'import visualization\.',
                r'from common\.',
                r'import common\.',
            ],
            'file_paths': [
                r'outputs/statistics/',
                r'outputs/metadata/',
                r'outputs/splits/',
                r'outputs/models/',
                r'outputs/consolidated_datasets/',
                r'outputs/emergency_saves/',
                r'outputs/incremental_saves/',
                r'outputs/zero_curtain_enhanced_cryogrid_physics_dataset\.parquet(?!.*part1_pinszc)',
                r'outputs/teacher_forcing_in_situ_database(?!.*part2_geocryoai)',
                r'configs/part2_config\.yaml',
                r'src/outputs/',
                r'src/physics_detection/',
                r'src/detection/',
                r'src/modeling/',
                r'src/data_ingestion/',
                r'src/processing/',
                r'src/transformations/',
                r'src/visualization/',
                r'src/common/',
            ]
        }
    
    def check_file(self, filepath: Path) -> List[Dict]:
        """Check a single file for old patterns."""
        file_issues = []
        
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Check import patterns
            for pattern in self.old_patterns['imports']:
                matches = re.finditer(pattern, content, re.MULTILINE)
                for match in matches:
                    # Get line number
                    line_num = content[:match.start()].count('\n') + 1
                    # Get the actual line
                    lines = content.split('\n')
                    line_content = lines[line_num - 1].strip()
                    
                    # Skip if it's in a comment
                    if line_content.startswith('#'):
                        continue
                    
                    # Skip if it's already updated to new path
                    if 'part1_physics_detection' in line_content or \
                       'part2_geocryoai' in line_content or \
                       'preprocessing' in line_content or \
                       'utils' in line_content:
                        continue
                    
                    file_issues.append({
                        'type': 'import',
                        'pattern': pattern,
                        'line': line_num,
                        'content': line_content
                    })
            
            # Check file path patterns
            for pattern in self.old_patterns['file_paths']:
                matches = re.finditer(pattern, content, re.MULTILINE)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    lines = content.split('\n')
                    line_content = lines[line_num - 1].strip()
                    
                    # Skip comments
                    if line_content.startswith('#'):
                        continue
                    
                    # Skip if in update_all_filepaths.py itself
                    if 'update_all_filepaths.py' in str(filepath):
                        continue
                    
                    file_issues.append({
                        'type': 'filepath',
                        'pattern': pattern,
                        'line': line_num,
                        'content': line_content
                    })
        
        except Exception as e:
            pass
        
        return file_issues
    
    def audit_all_files(self):
        """Audit all relevant files in repository."""
        print("=" * 80)
        print("COMPREHENSIVE PATH AUDIT")
        print("=" * 80)
        print()
        
        # File types to check
        extensions = ['*.py', '*.sh', '*.yaml', '*.yml', '*.md', '*.txt', '*.json']
        
        all_files = []
        for ext in extensions:
            all_files.extend(self.root_dir.rglob(ext))
        
        # Filter out .git and __pycache__
        all_files = [
            f for f in all_files
            if '.git' not in str(f) and '__pycache__' not in str(f)
        ]
        
        print(f"Checking {len(all_files)} files...")
        print()
        
        files_with_issues = {}
        
        for filepath in sorted(all_files):
            issues = self.check_file(filepath)
            if issues:
                rel_path = filepath.relative_to(self.root_dir)
                files_with_issues[str(rel_path)] = issues
        
        # Report results
        print("=" * 80)
        print("AUDIT RESULTS")
        print("=" * 80)
        print()
        
        if not files_with_issues:
            print("✅ NO ISSUES FOUND - All paths are up to date!")
            print()
            return True
        
        print(f"⚠️  FOUND ISSUES IN {len(files_with_issues)} FILES")
        print()
        
        for filepath, issues in files_with_issues.items():
            print(f"\n{filepath}:")
            print("-" * 80)
            
            for issue in issues:
                print(f"  Line {issue['line']} ({issue['type']}):")
                print(f"    Pattern: {issue['pattern']}")
                print(f"    Content: {issue['content'][:100]}")
                print()
        
        print("=" * 80)
        print(f"SUMMARY: {len(files_with_issues)} files need attention")
        print("=" * 80)
        
        return False

def main():
    root_dir = Path(__file__).parent.parent
    auditor = PathAuditor(root_dir)
    clean = auditor.audit_all_files()
    
    if not clean:
        print("\n⚠️  Run the fixes and re-audit before committing!")
        exit(1)
    else:
        print("✅ Repository is clean - ready to commit!")
        exit(0)

if __name__ == "__main__":
    main()