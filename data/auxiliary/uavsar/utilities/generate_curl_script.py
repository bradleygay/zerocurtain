import re
from collections import defaultdict
import os

def extract_urls_from_sh(sh_content):
    """
    Extract all URLs from the bash script's wget commands.
    """
    urls_by_flightline = defaultdict(list)
    
    lines = sh_content.split('\n')
    current_fl = None
    
    for line in lines:
        mkdir_match = re.search(r'mkdir -p nisar_downloads/([a-zA-Z]{6}_[\d_]+)', line)
        if mkdir_match:
            current_fl = mkdir_match.group(1)
            continue
        
        echo_match = re.search(r"echo 'Downloading ([a-zA-Z]{6}_[\d_]+) files", line)
        if echo_match:
            current_fl = echo_match.group(1)
            continue
        
        if current_fl and 'grep -o' in line:
            url_match = re.search(r"'(https://[^']+)'", line)
            if url_match:
                url_pattern = url_match.group(1)
                urls_by_flightline[current_fl].append(('jpl_pattern', url_pattern))
    
    return urls_by_flightline

def extract_urls_from_txt(txt_content):
    """
    Extract URLs from the txt file - keep full URL intact.
    """
    urls_by_flightline = defaultdict(list)
    
    for line in txt_content.strip().split('\n'):
        if not line.startswith('wget '):
            continue
        
        url = line.replace('wget ', '').strip()
        
        # Pattern 1: ASF - extract the directory name after UA_
        match1 = re.search(r'uavsar\.asf\.alaska\.edu/UA_([a-zA-Z]{6}_[^/]+)/', url)
        if match1:
            fl_id = match1.group(1)
            urls_by_flightline[fl_id].append(('asf_direct', url))
            continue
        
        # Pattern 2 & 3: JPL - extract directory name after Release*/
        match2 = re.search(r'(?:downloaduav|uavsar)\.jpl\.nasa\.gov/Release[^/]*/([a-zA-Z]{6}_[^/]+)/', url)
        if match2:
            fl_id = match2.group(1)
            urls_by_flightline[fl_id].append(('jpl_direct', url))
    
    return urls_by_flightline

def generate_curl_script(sh_file, txt_file, output_file='uavsar_download_curl.sh'):
    """
    Generate a curl-based download script with proper ASF authentication.
    """
    with open(sh_file, 'r') as f:
        sh_content = f.read()
    
    with open(txt_file, 'r') as f:
        txt_content = f.read()
    
    print("Extracting URLs from nisar_download.sh...")
    sh_urls = extract_urls_from_sh(sh_content)
    print(f"  Found {len(sh_urls)} flight lines")
    
    print("Extracting URLs from uavsar_wget_parallel.txt...")
    txt_urls = extract_urls_from_txt(txt_content)
    print(f"  Found {len(txt_urls)} flight lines")
    
    # Merge
    all_urls = {}
    for fl, urls in sh_urls.items():
        all_urls[fl] = {'urls': urls, 'source': 'sh'}
    
    for fl, urls in txt_urls.items():
        if fl in all_urls:
            all_urls[fl]['urls'].extend(urls)
            all_urls[fl]['source'] = 'both'
        else:
            all_urls[fl] = {'urls': urls, 'source': 'txt'}
    
    sh_only = sum(1 for v in all_urls.values() if v['source'] == 'sh')
    txt_only = sum(1 for v in all_urls.values() if v['source'] == 'txt')
    both = sum(1 for v in all_urls.values() if v['source'] == 'both')
    
    print(f"\nMerge results:")
    print(f"  From sh only: {sh_only}")
    print(f"  From txt only: {txt_only}")
    print(f"  From both: {both}")
    print(f"  Total unique: {len(all_urls)}")
    
    # Generate script
    script = ['#!/bin/bash']
    script.append('# UAVSAR Download Script - curl version with ASF authentication')
    script.append('# Consolidated from nisar_download.sh and uavsar_wget_parallel.txt')
    script.append('#')
    script.append('# IMPORTANT: ASF data requires [RESEARCH_INSTITUTION] Earthdata authentication')
    script.append('# Create ~/.netrc with your credentials:')
    script.append('#   machine urs.earthdata.[RESEARCH_INSTITUTION_DOMAIN]')
    script.append('#       login YOUR_USERNAME')
    script.append('#       password YOUR_PASSWORD')
    script.append('#   chmod 600 ~/.netrc')
    script.append('')
    script.append('set -e')
    script.append('BASE_DIR="nisar_downloads"')
    script.append('COOKIE_FILE="$HOME/.urs_cookies"')
    script.append('mkdir -p "$BASE_DIR"')
    script.append('')
    script.append('# Check for .netrc file')
    script.append('if [ ! -f ~/.netrc ]; then')
    script.append('    echo "ERROR: ~/.netrc file not found!"')
    script.append('    echo "Please create ~/.netrc with your [RESEARCH_INSTITUTION] Earthdata credentials"')
    script.append('    exit 1')
    script.append('fi')
    script.append('')
    script.append('# Download function for ASF URLs (requires auth with cookies)')
    script.append('download_asf_file() {')
    script.append('    local url="$1"')
    script.append('    local output_path="$2"')
    script.append('    local filename=$(basename "$output_path")')
    script.append('    ')
    script.append('    if [ -f "$output_path" ]; then')
    script.append('        echo "   Skip: $filename (exists)"')
    script.append('        return 0')
    script.append('    fi')
    script.append('    ')
    script.append('    echo "   Downloading: $filename"')
    script.append('    # Use netrc for auth, cookie jar for session, follow redirects with limit')
    script.append('    if curl -n -c "$COOKIE_FILE" -b "$COOKIE_FILE" -L --max-redirs 10 -f -s -S --create-dirs -o "$output_path" "$url" 2>&1; then')
    script.append('        echo "   Complete: $filename"')
    script.append('        return 0')
    script.append('    else')
    script.append('        local exit_code=$?')
    script.append('        echo "   Failed: $filename (curl exit code: $exit_code)"')
    script.append('        rm -f "$output_path"')
    script.append('        # If redirect error, try once more with fresh cookies')
    script.append('        if [ $exit_code -eq 47 ]; then')
    script.append('            echo "  ↻ Retry with fresh authentication..."')
    script.append('            rm -f "$COOKIE_FILE"')
    script.append('            if curl -n -c "$COOKIE_FILE" -b "$COOKIE_FILE" -L --max-redirs 10 -f -s -S --create-dirs -o "$output_path" "$url" 2>&1; then')
    script.append('                echo "   Complete: $filename (retry successful)"')
    script.append('                return 0')
    script.append('            fi')
    script.append('        fi')
    script.append('        return 1')
    script.append('    fi')
    script.append('}')
    script.append('')
    script.append('# Download function for JPL URLs (no auth required)')
    script.append('download_jpl_file() {')
    script.append('    local url="$1"')
    script.append('    local output_path="$2"')
    script.append('    local filename=$(basename "$output_path")')
    script.append('    ')
    script.append('    if [ -f "$output_path" ]; then')
    script.append('        echo "   Skip: $filename (exists)"')
    script.append('        return 0')
    script.append('    fi')
    script.append('    ')
    script.append('    echo "   Downloading: $filename"')
    script.append('    if curl -L -f -s -S --create-dirs -o "$output_path" "$url" 2>&1; then')
    script.append('        echo "   Complete: $filename"')
    script.append('        return 0')
    script.append('    else')
    script.append('        echo "   Failed: $filename"')
    script.append('        rm -f "$output_path"')
    script.append('        return 1')
    script.append('    fi')
    script.append('}')
    script.append('')
    script.append('# JPL pattern download - tries multiple release versions')
    script.append('download_jpl_pattern() {')
    script.append('    local url_pattern="$1"')
    script.append('    local output_path="$2"')
    script.append('    local filename=$(basename "$output_path")')
    script.append('    ')
    script.append('    if [ -f "$output_path" ]; then')
    script.append('        echo "   Skip: $filename (exists)"')
    script.append('        return 0')
    script.append('    fi')
    script.append('    ')
    script.append('    local file_path=$(echo "$url_pattern" | sed -E "s|.*/Release\[.*\]\*/||")')
    script.append('    ')
    script.append('    for release in Release24 Release23 Release22 Release21 Release20 Release19 Release18 Release17 Release16 Release15; do')
    script.append('        local url="https://downloaduav.jpl.[RESEARCH_INSTITUTION_DOMAIN]/${release}/${file_path}"')
    script.append('        if curl -L -f -s -S --head "$url" > /dev/null 2>&1; then')
    script.append('            echo "   Downloading: $filename (${release})"')
    script.append('            if curl -L -f -s -S --create-dirs -o "$output_path" "$url" 2>&1; then')
    script.append('                echo "   Complete: $filename"')
    script.append('                return 0')
    script.append('            fi')
    script.append('        fi')
    script.append('    done')
    script.append('    ')
    script.append('    echo "   Failed: $filename (not found in any release)"')
    script.append('    rm -f "$output_path"')
    script.append('    return 1')
    script.append('}')
    script.append('')
    script.append(f'echo "Starting download of {len(all_urls)} flight lines..."')
    script.append('echo "Cookie file: $COOKIE_FILE"')
    script.append('echo ""')
    script.append('')
    
    # Generate download commands
    for idx, (fl_id, data) in enumerate(sorted(all_urls.items()), 1):
        script.append(f'# Flight line {idx}/{len(all_urls)}: {fl_id}')
        script.append(f'echo ""')
        script.append(f'echo "Flight line {idx}/{len(all_urls)}: {fl_id}"')
        script.append(f'mkdir -p "$BASE_DIR/{fl_id}"')
        script.append('')
        
        for url_type, url_data in data['urls']:
            filename = os.path.basename(url_data) if url_type != 'jpl_pattern' else re.search(r'/([^/]+)$', url_data).group(1) if re.search(r'/([^/]+)$', url_data) else 'unknown'
            
            if url_type == 'asf_direct':
                script.append(f'download_asf_file "{url_data}" "$BASE_DIR/{fl_id}/{filename}"')
            elif url_type == 'jpl_direct':
                script.append(f'download_jpl_file "{url_data}" "$BASE_DIR/{fl_id}/{filename}"')
            elif url_type == 'jpl_pattern':
                script.append(f'download_jpl_pattern "{url_data}" "$BASE_DIR/{fl_id}/{filename}"')
        
        script.append('')
    
    script.append('echo ""')
    script.append('echo ""')
    script.append('echo "Download complete!"')
    script.append(f'echo "Files saved to: $BASE_DIR"')
    script.append('echo "Cleaning up cookies..."')
    script.append('rm -f "$COOKIE_FILE"')
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(script))
    
    os.chmod(output_file, 0o755)
    
    return output_file

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 2:
        sh_file = sys.argv[1]
        txt_file = sys.argv[2]
    else:
        sh_file = "/Users/[USER]/arctic_zero_curtain_pipeline/data/auxiliary/uavsar/nisar_download.sh"
        txt_file = "/Users/[USER]/arctic_zero_curtain_pipeline/data/auxiliary/uavsar/uavsar_wget_parallel.txt"
    
    output = generate_curl_script(sh_file, txt_file)
    print(f"\n{'='*60}")
    print(f"Generated: {output}")
    print(f"\nEnsure ~/.netrc is configured, then execute:")
    print(f"  ./{output}")
    print(f"{'='*60}")

# python generate_curl_script.py nisar_download.sh uavsar_wget_parallel.txt
# ./uavsar_download_curl.sh