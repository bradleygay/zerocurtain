import re
from collections import defaultdict

def extract_flightlines_from_sh(sh_content):
    """
    Extract flight line identifiers and their wget commands from bash script.
    """
    flightlines = {}
    current_fl = None
    
    lines = sh_content.split('\n')
    
    for line in lines:
        # Check for flight line directory creation
        mkdir_match = re.search(r'mkdir -p nisar_downloads/([a-zA-Z]{6}_[\d_]+)', line)
        if mkdir_match:
            current_fl = mkdir_match.group(1)
            if current_fl not in flightlines:
                flightlines[current_fl] = []
            continue
        
        # Check for echo statements indicating new flight line
        echo_match = re.search(r"echo 'Downloading ([a-zA-Z]{6}_[\d_]+) files\.\.\.'", line)
        if echo_match:
            current_fl = echo_match.group(1)
            if current_fl not in flightlines:
                flightlines[current_fl] = []
            continue
        
        # Extract wget commands
        if current_fl and 'wget' in line and 'https://' in line:
            # Extract the full wget command
            flightlines[current_fl].append(line.strip())
    
    return flightlines

def extract_flightlines_from_txt(txt_content):
    """
    Extract flight line identifiers and URLs from wget text file.
    Handles three URL patterns:
    1. https://uavsar.asf.alaska.edu/UA_[identifier]/
    2. https://downloaduav.jpl.[RESEARCH_INSTITUTION_DOMAIN]/Release**/[identifier]/
    3. https://uavsar.jpl.[RESEARCH_INSTITUTION_DOMAIN]/Release**/[identifier]/
    """
    flightlines = defaultdict(list)
    
    lines = txt_content.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line.startswith('wget'):
            continue
        
        # Extract URL
        url_match = re.search(r'https?://[^\s]+', line)
        if not url_match:
            continue
        
        url = url_match.group(0)
        
        # Try Pattern 1: ASF URLs with UA_ prefix
        match1 = re.search(r'uavsar\.asf\.alaska\.edu/UA_([a-zA-Z]{6}_[\d\-_]+?)/', url)
        if match1:
            fl_id = match1.group(1)
            flightlines[fl_id].append(url)
            continue
        
        # Try Pattern 2: JPL downloaduav URLs
        match2 = re.search(r'downloaduav\.jpl\.nasa\.gov/Release[^/]*/([a-zA-Z]{6}_[\d_]+?)(?:_[A-Z]\d+)?(?:_[A-Z]{2})?(?:_\d+)?/', url)
        if match2:
            fl_id = match2.group(1)
            flightlines[fl_id].append(url)
            continue
        
        # Try Pattern 3: JPL uavsar URLs
        match3 = re.search(r'uavsar\.jpl\.nasa\.gov/Release[^/]*/([a-zA-Z]{6}_[\d_]+?)(?:_[A-Z]\d+)?(?:_[A-Z]{2})?(?:_\d+)?/', url)
        if match3:
            fl_id = match3.group(1)
            flightlines[fl_id].append(url)
            continue
    
    return flightlines

def merge_and_generate_script(sh_file_path, txt_file_path, output_file='uavsar_download_consolidated.sh'):
    """
    Merge flight lines from both nisar_download.sh and uavsar_wget_parallel.txt
    into a single consolidated download script.
    """
    # Read both files
    with open(sh_file_path, 'r') as f:
        sh_content = f.read()
    
    with open(txt_file_path, 'r') as f:
        txt_content = f.read()
    
    # Extract flight lines from both sources
    print("Extracting flight lines from nisar_download.sh...")
    sh_flightlines = extract_flightlines_from_sh(sh_content)
    print(f"  Found {len(sh_flightlines)} flight lines")
    
    print("Extracting flight lines from uavsar_wget_parallel.txt...")
    txt_flightlines = extract_flightlines_from_txt(txt_content)
    print(f"  Found {len(txt_flightlines)} flight lines")
    
    # Merge flight lines (txt takes precedence for duplicates, or we can combine)
    all_flightlines = {}
    
    # Add all from sh script first
    for fl, commands in sh_flightlines.items():
        all_flightlines[fl] = {
            'source': 'nisar_download.sh',
            'commands': commands
        }
    
    # Add from txt file (mark overlaps)
    for fl, urls in txt_flightlines.items():
        if fl in all_flightlines:
            # Flight line exists in both - merge commands
            all_flightlines[fl]['source'] = 'both'
            all_flightlines[fl]['commands'].extend(urls)
        else:
            # New flight line from txt
            all_flightlines[fl] = {
                'source': 'uavsar_wget_parallel.txt',
                'commands': urls
            }
    
    # Report statistics
    sh_only = sum(1 for fl in all_flightlines.values() if fl['source'] == 'nisar_download.sh')
    txt_only = sum(1 for fl in all_flightlines.values() if fl['source'] == 'uavsar_wget_parallel.txt')
    both = sum(1 for fl in all_flightlines.values() if fl['source'] == 'both')
    
    print(f"\nMerge Statistics:")
    print(f"  Flight lines only in nisar_download.sh: {sh_only}")
    print(f"  Flight lines only in uavsar_wget_parallel.txt: {txt_only}")
    print(f"  Flight lines in both files: {both}")
    print(f"  Total unique flight lines: {len(all_flightlines)}")
    
    # Generate consolidated script
    script_lines = [
        '#!/bin/bash',
        '# UAVSAR Consolidated Download Script',
        '# Merged from nisar_download.sh and uavsar_wget_parallel.txt',
        f'# Total flight lines: {len(all_flightlines)}',
        f'# Generated: {__import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        '',
        'set -e  # Exit on error',
        'set -u  # Exit on undefined variable',
        '',
        'BASE_DIR="nisar_downloads"',
        'mkdir -p "$BASE_DIR"',
        '',
        'echo "Starting UAVSAR consolidated data download..."',
        f'echo "Total flight lines to process: {len(all_flightlines)}"',
        'echo ""',
        ''
    ]
    
    # Sort flight lines for organized output
    sorted_flightlines = sorted(all_flightlines.keys())
    
    for idx, fl_id in enumerate(sorted_flightlines, 1):
        fl_data = all_flightlines[fl_id]
        commands = fl_data['commands']
        source = fl_data['source']
        
        script_lines.append(f'# Flight line {idx}/{len(sorted_flightlines)}: {fl_id}')
        script_lines.append(f'# Source: {source}')
        script_lines.append(f'echo "Downloading {fl_id} ({len(commands)} commands)..."')
        script_lines.append(f'mkdir -p "$BASE_DIR/{fl_id}"')
        script_lines.append('')
        
        for cmd in commands:
            # Normalize command format
            if cmd.startswith('wget'):
                # Already a wget command from sh script - use as is
                script_lines.append(cmd)
            else:
                # URL from txt file - create wget command
                script_lines.append(f'wget -P "$BASE_DIR/{fl_id}" -nc {cmd}')
        
        script_lines.append('')
        script_lines.append(f'echo "Completed: {fl_id}"')
        script_lines.append('echo ""')
        script_lines.append('')
    
    script_lines.append('echo "All downloads complete!"')
    script_lines.append(f'echo "Data saved to: $BASE_DIR"')
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(script_lines))
    
    # Make executable
    import os
    os.chmod(output_file, 0o755)
    
    return output_file, all_flightlines, (sh_only, txt_only, both)

def main(sh_file, txt_file):
    print("="*70)
    print("UAVSAR Consolidated Download Script Generator")
    print("Merging nisar_download.sh + uavsar_wget_parallel.txt")
    print("="*70)
    print()
    
    output_file, all_flightlines, stats = merge_and_generate_script(sh_file, txt_file)
    
    sh_only, txt_only, both = stats
    
    print(f"\n{'GENERATION COMPLETE':^70}")
    print("-"*70)
    print(f"Output file: {output_file}")
    print(f"Total flight lines: {len(all_flightlines)}")
    print(f"  - From nisar_download.sh only: {sh_only}")
    print(f"  - From uavsar_wget_parallel.txt only: {txt_only}")
    print(f"  - From both sources: {both}")
    print(f"\nTotal download commands: {sum(len(fl['commands']) for fl in all_flightlines.values())}")
    print("\nTo download all data, execute:")
    print(f"  bash {output_file}")
    print("="*70)
    print()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 2:
        sh_file = sys.argv[1]
        txt_file = sys.argv[2]
    else:
        sh_file = "/Users/[USER]/arctic_zero_curtain_pipeline/data/auxiliary/uavsar/nisar_download.sh"
        txt_file = "/Users/[USER]/arctic_zero_curtain_pipeline/data/auxiliary/uavsar/uavsar_wget_parallel.txt"
    
    main(sh_file, txt_file)
