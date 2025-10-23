#!/usr/bin/env python3
"""
Automated Script Integration for Google Drive Data Access

Automatically updates all Python scripts to use get_data_file() for large files.
Scans LARGE_FILES_MAPPING to identify files requiring remote access.

Usage:
    python automate_data_access_integration.py --dry-run  # Preview changes
    python automate_data_access_integration.py --apply    # Apply changes
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple, Set
import argparse

# Import large files mapping
sys.path.insert(0, str(Path(__file__).parent))
from setup_gdrive_placeholders import LARGE_FILES_MAPPING


class DataAccessIntegrator:
    """Automatically integrate get_data_file() into Python scripts."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.large_file_patterns = self._extract_file_patterns()
        self.changes_made = []
        
    def _extract_file_patterns(self) -> Set[str]:
        """Extract file path patterns from LARGE_FILES_MAPPING."""
        patterns = set()
        
        for filepath in LARGE_FILES_MAPPING.keys():
            # Add full path
            patterns.add(filepath)
            
            # Add just filename (for cases where path varies)
            patterns.add(Path(filepath).name)
        
        return patterns
    
    def _needs_import(self, content: str) -> bool:
        """Check if script already imports get_data_file."""
        import_patterns = [
            r'from\s+src\.utils\.data_access\s+import\s+get_data_file',
            r'from\s+src\.utils\.data_access\s+import\s+.*get_data_file',
            r'import\s+src\.utils\.data_access',
        ]
        
        return not any(re.search(pattern, content) for pattern in import_patterns)
    
    def _add_import(self, content: str) -> str:
        """Add get_data_file import to script."""
        # Find where to insert import (after existing imports)
        lines = content.split('\n')
        insert_idx = 0
        
        # Skip shebang and docstring
        in_docstring = False
        for i, line in enumerate(lines):
            if line.strip().startswith('"""') or line.strip().startswith("'''"):
                in_docstring = not in_docstring
            
            if not in_docstring and not line.startswith('#!') and line.strip():
                if line.startswith('import ') or line.startswith('from '):
                    insert_idx = i + 1
                elif insert_idx > 0:
                    break
        
        # Insert import
        import_statement = "from src.utils.data_access import get_data_file"
        
        # Check if there's already a blank line
        if insert_idx < len(lines) and lines[insert_idx].strip():
            import_statement += '\n'
        
        lines.insert(insert_idx, import_statement)
        
        return '\n'.join(lines)
    
    def _find_file_loads(self, content: str) -> List[Tuple[str, str, str]]:
        """
        Find file loading statements that should use get_data_file().
        
        Returns:
            List of (full_match, filepath, loading_function) tuples
        """
        matches = []
        
        # Patterns for file loading
        patterns = [
            # pd.read_parquet('path')
            (r"(pd\.read_parquet\s*\(\s*['\"]([^'\"]+)['\"])", 'read_parquet'),
            # pd.read_csv('path')
            (r"(pd\.read_csv\s*\(\s*['\"]([^'\"]+)['\"])", 'read_csv'),
            # torch.load('path')
            (r"(torch\.load\s*\(\s*['\"]([^'\"]+)['\"])", 'torch.load'),
            # xr.open_dataset('path')
            (r"(xr\.open_dataset\s*\(\s*['\"]([^'\"]+)['\"])", 'xr.open_dataset'),
            # gpd.read_file('path')
            (r"(gpd\.read_file\s*\(\s*['\"]([^'\"]+)['\"])", 'gpd.read_file'),
            # open('path')
            (r"(open\s*\(\s*['\"]([^'\"]+\.(?:parquet|pth|nc|tif|shp))['\"])", 'open'),
        ]
        
        for pattern, func_name in patterns:
            for match in re.finditer(pattern, content):
                full_match = match.group(1)
                filepath = match.group(2)
                
                # Check if this file is in large files mapping
                if self._is_large_file(filepath):
                    matches.append((full_match, filepath, func_name))
        
        return matches
    
    def _is_large_file(self, filepath: str) -> bool:
        """Check if filepath corresponds to a large file."""
        # Direct match
        if filepath in self.large_file_patterns:
            return True
        
        # Filename match
        if Path(filepath).name in self.large_file_patterns:
            return True
        
        # Pattern match (ends with known large file)
        for pattern in self.large_file_patterns:
            if filepath.endswith(pattern) or pattern.endswith(Path(filepath).name):
                return True
        
        return False
    
    def _replace_file_load(self, match_tuple: Tuple[str, str, str]) -> str:
        """Generate replacement code using get_data_file()."""
        full_match, filepath, func_name = match_tuple
        
        # Generate replacement based on function type
        if func_name == 'read_parquet':
            return f"pd.read_parquet(get_data_file('{filepath}'))"
        elif func_name == 'read_csv':
            return f"pd.read_csv(get_data_file('{filepath}'))"
        elif func_name == 'torch.load':
            return f"torch.load(get_data_file('{filepath}')"
        elif func_name == 'xr.open_dataset':
            return f"xr.open_dataset(get_data_file('{filepath}'))"
        elif func_name == 'gpd.read_file':
            return f"gpd.read_file(get_data_file('{filepath}'))"
        elif func_name == 'open':
            return f"open(get_data_file('{filepath}'))"
        else:
            return full_match  # No change
    
    def process_file(self, filepath: Path, dry_run: bool = True) -> bool:
        """
        Process a single Python file.
        
        Args:
            filepath: Path to Python file
            dry_run: If True, only show changes without applying
        
        Returns:
            True if changes were made/would be made
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                original_content = f.read()
        except Exception as e:
            print(f"   Error reading {filepath}: {e}")
            return False
        
        content = original_content
        changes = []
        
        # Step 1: Find file loads that need updating
        file_loads = self._find_file_loads(content)
        
        if not file_loads:
            return False  # No changes needed
        
        # Step 2: Add import if needed
        if self._needs_import(content):
            content = self._add_import(content)
            changes.append("Added get_data_file import")
        
        # Step 3: Replace file loads
        for match_tuple in file_loads:
            full_match, filepath_str, func_name = match_tuple
            replacement = self._replace_file_load(match_tuple)
            
            content = content.replace(full_match, replacement)
            changes.append(f"Updated: {func_name}('{filepath_str}')")
        
        # Step 4: Apply or preview
        if dry_run:
            print(f"\n {filepath.relative_to(self.project_root)}")
            print(f"   Changes to make:")
            for change in changes:
                print(f"     • {change}")
        else:
            # Write updated content
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"\n {filepath.relative_to(self.project_root)}")
            for change in changes:
                print(f"   • {change}")
            
            self.changes_made.append(filepath)
        
        return True
    
    def process_directory(self, directory: Path, dry_run: bool = True):
        """Process all Python files in directory recursively."""
        python_files = directory.rglob('*.py')
        
        files_processed = 0
        files_changed = 0
        
        for py_file in python_files:
            # Skip setup and automation scripts
            if py_file.name in ['setup_gdrive_placeholders.py', 
                                'automate_data_access_integration.py',
                                'predownload_all_data.py']:
                continue
            
            files_processed += 1
            
            if self.process_file(py_file, dry_run=dry_run):
                files_changed += 1
        
        return files_processed, files_changed


def main():
    """Main execution."""
    parser = argparse.ArgumentParser(
        description="Automatically integrate get_data_file() into Python scripts"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without applying them'
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help='Apply changes to files'
    )
    parser.add_argument(
        '--directory',
        type=str,
        default='src',
        help='Directory to process (default: src/)'
    )
    
    args = parser.parse_args()
    
    if not args.dry_run and not args.apply:
        print("Error: Must specify either --dry-run or --apply")
        parser.print_help()
        sys.exit(1)
    
    project_root = Path(__file__).parent
    target_dir = project_root / args.directory
    
    if not target_dir.exists():
        print(f"Error: Directory not found: {target_dir}")
        sys.exit(1)
    
    print("="*70)
    print("AUTOMATED DATA ACCESS INTEGRATION")
    print("="*70)
    print(f"Project root: {project_root}")
    print(f"Target directory: {target_dir}")
    print(f"Mode: {'DRY RUN (preview only)' if args.dry_run else 'APPLY CHANGES'}")
    print(f"Large files tracked: {len(LARGE_FILES_MAPPING)}")
    print("="*70)
    
    if args.dry_run:
        print("\n  DRY RUN MODE - No files will be modified")
        print("Review changes below, then run with --apply to make changes\n")
    
    # Process files
    integrator = DataAccessIntegrator(project_root)
    files_processed, files_changed = integrator.process_directory(
        target_dir, 
        dry_run=args.dry_run
    )
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Files processed: {files_processed}")
    print(f"Files {'that would be' if args.dry_run else ''} changed: {files_changed}")
    
    if args.dry_run and files_changed > 0:
        print("\n Review changes above")
        print(" Run with --apply to make changes:")
        print(f"    python {Path(__file__).name} --apply")
    elif not args.dry_run and files_changed > 0:
        print(f"\n Successfully updated {files_changed} files")
        print("\n Next steps:")
        print("1. Review changes with: git diff")
        print("2. Test updated scripts")
        print("3. Commit changes:")
        print(f"   git add {args.directory}/")
        print('   git commit -m "feat: integrate Google Drive data access layer"')
    else:
        print("\n No changes needed - all scripts already use get_data_file()")
    
    print("="*70)


if __name__ == "__main__":
    main()
