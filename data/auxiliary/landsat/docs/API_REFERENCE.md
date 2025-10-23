# Landsat Processing Pipeline - API Reference

**Version**: 1.0.0  
**Last Updated**: 2025-01-20

Complete API documentation for all modules, classes, and functions in the Landsat processing pipeline.

---

## Table of Contents

1. [Core Modules](#core-modules)
2. [Configuration Classes](#configuration-classes)
3. [Earth Engine Interface](#earth-engine-interface)
4. [Processing Functions](#processing-functions)
5. [Utility Functions](#utility-functions)
6. [Data Structures](#data-structures)

---

## Core Modules

### `landsat_downloader.py`

Primary module for downloading Landsat 8/9 Collection 2 Level-2 data from Google Earth Engine.

#### Module-Level Constants
```python
Config.MAX_CONCURRENT_PROCESSES  # int: Maximum parallel processes
Config.MAX_THREADS_PER_PROCESS   # int: Maximum threads per process
Config.SCENE_BATCH_SIZE          # int: Scenes per batch
Config.POINTS_PER_SCENE          # int: Sample points per scene
Config.CSV_BATCH_SIZE            # int: Rows per CSV batch
Config.TILE_SCALE                # int: Earth Engine tile scale
Config.RETRY_LIMIT               # int: Maximum retry attempts
Config.RETRY_DELAY               # int: Initial retry delay (seconds)
```

#### Main Functions

##### `main()`
```python
def main() -> int
```

Main execution function for Landsat download pipeline.

**Arguments**: None (uses command-line arguments)

**Command-Line Arguments**:
- `--start-date` (str): Start date in YYYY-MM-DD format. Default: '2015-03-30'
- `--end-date` (str): End date in YYYY-MM-DD format. Default: '2024-12-31'
- `--cloud-threshold` (int): Maximum cloud cover percentage. Default: 10
- `--output-dir` (str): Output directory path. Default: '~/arctic_zero_curtain_pipeline/data/auxiliary/landsat'
- `--max-processes` (int): Maximum concurrent processes. Default: 8
- `--force-restart` (flag): Ignore checkpoints and restart processing
- `--debug` (flag): Enable debug logging

**Returns**: int - Exit code (0 = success, 1 = failure)

**Example**:
```bash
python landsat_downloader.py \
    --start-date 2020-01-01 \
    --end-date 2023-12-31 \
    --cloud-threshold 15 \
    --max-processes 4
```

---

### `landsat_gap_filler.py`

Module for identifying and filling spatial gaps in Landsat coverage.

#### Main Functions

##### `analyze_coverage()`
```python
def analyze_coverage(data_file: Path, grid_size: float = 5.0) -> Dict
```

Analyze spatial coverage of Landsat dataset to identify gaps.

**Parameters**:
- `data_file` (Path): Path to input parquet file
- `grid_size` (float): Grid cell size in degrees. Default: 5.0

**Returns**: Dict containing:
```python
{
    'coverage_percentage': float,      # Percentage of domain covered
    'total_cells': int,                # Total grid cells
    'filled_cells': int,               # Cells with data
    'empty_cells': int,                # Cells without data
    'gap_cells': List[Dict],           # List of gap cell definitions
    'boundary_issues': Dict[str, bool], # Boundary coverage flags
    'lon_range': List[float],          # [min, max] longitude
    'lat_range': List[float]           # [min, max] latitude
}
```

**Example**:
```python
from pathlib import Path
from landsat_gap_filler import analyze_coverage

data_file = Path('raw/landsat_arctic_data.parquet')
coverage = analyze_coverage(data_file, grid_size=5.0)

print(f"Coverage: {coverage['coverage_percentage']:.2f}%")
print(f"Gap cells: {len(coverage['gap_cells'])}")
```

##### `create_gap_regions()`
```python
def create_gap_regions(coverage_analysis: Dict) -> List[Dict]
```

Generate region definitions for filling identified gaps.

**Parameters**:
- `coverage_analysis` (Dict): Output from `analyze_coverage()`

**Returns**: List[Dict] - List of region dictionaries:
```python
[
    {
        'name': str,           # Region identifier
        'coords': List[float], # [lon_min, lat_min, lon_max, lat_max]
        'type': str           # Region type: 'empty_cell', 'boundary', etc.
    },
    ...
]
```

**Example**:
```python
from landsat_gap_filler import analyze_coverage, create_gap_regions

coverage = analyze_coverage(data_file)
gap_regions = create_gap_regions(coverage)

for region in gap_regions[:5]:
    print(f"{region['name']}: {region['type']}")
```

---

### `landsat_combiner.py`

Module for merging multiple Landsat datasets.

#### Main Functions

##### `combine_datasets()`
```python
def combine_datasets(
    original_file: Path,
    gap_file: Path,
    swath_gap_file: Path,
    output_file: Path
) -> pd.DataFrame
```

Combine original, gap-filled, and swath gap-filled datasets.

**Parameters**:
- `original_file` (Path): Path to original dataset
- `gap_file` (Path): Path to gap-filled dataset
- `swath_gap_file` (Path): Path to swath gap-filled dataset
- `output_file` (Path): Path for combined output

**Returns**: pd.DataFrame - Combined dataset with duplicates removed

**Process**:
1. Load all three datasets
2. Concatenate into single DataFrame
3. Remove duplicates based on (longitude, latitude, scene_id)
4. Save to output file
5. Return combined DataFrame

**Example**:
```python
from pathlib import Path
from landsat_combiner import combine_datasets

combined = combine_datasets(
    Path('raw/landsat_arctic_data.parquet'),
    Path('gaps_filled/raw/landsat_gaps_filled.parquet'),
    Path('swath_gaps_filled/landsat_arctic_swath_gaps_data.parquet'),
    Path('final/landsat_complete.parquet')
)

print(f"Combined dataset: {len(combined):,} points")
```

---

## Configuration Classes

### `Config`

Central configuration class containing all pipeline parameters.

#### Class Attributes
```python
class Config:
    # System Resources
    MAX_CONCURRENT_PROCESSES: int = min(8, multiprocessing.cpu_count())
    MAX_THREADS_PER_PROCESS: int = 8
    
    # Processing Parameters
    SCENE_BATCH_SIZE: int = 64
    POINTS_PER_SCENE: int = 100
    CSV_BATCH_SIZE: int = 5000
    TILE_SCALE: int = 16
    
    # Retry Configuration
    RETRY_LIMIT: int = 10
    RETRY_DELAY: int = 8
    
    # Spatial Domain
    LON_MIN: float = -180.0
    LON_MAX: float = 180.0
    LAT_MIN: float = 45.0
    LAT_MAX: float = 90.0
    
    # Grid Resolution
    LON_STEP: float = 10.0
    LAT_STEP: float = 5.0
    
    # Earth Engine
    EE_PROJECT: str = 'circumarcticzerocurtain'
    LANDSAT_COLLECTION: str = 'LANDSAT/LC08/C02/T1_L2'
    LANDSAT_BANDS: List[str] = ['SR_B2', 'SR_B3', 'SR_B4', 'ST_B10']
    BAND_NAMES: List[str] = ['B2', 'B3', 'B4', 'B10']
```

#### Class Methods

##### `get_output_dirs()`
```python
@classmethod
def get_output_dirs(cls, base_dir: str) -> Dict[str, Path]
```

Generate output directory structure.

**Parameters**:
- `base_dir` (str): Base directory path

**Returns**: Dict[str, Path] - Dictionary mapping directory names to Path objects:
```python
{
    'raw': Path,
    'checkpoints': Path,
    'temp_csv': Path,
    'analysis': Path,
    'logs': Path,
    'processed': Path,
    'gaps_filled': Path,
    'swath_gaps_filled': Path,
    'final': Path
}
```

**Example**:
```python
from pathlib import Path

base = Path('~/arctic_zero_curtain_pipeline/data/auxiliary/landsat').expanduser()
dirs = Config.get_output_dirs(base)

print(f"Raw data directory: {dirs['raw']}")
print(f"Logs directory: {dirs['logs']}")
```

---

## Earth Engine Interface

### Authentication and Initialization

#### `initialize_earth_engine()`
```python
def initialize_earth_engine(project: str, max_retries: int = 3) -> bool
```

Initialize Google Earth Engine with retry logic.

**Parameters**:
- `project` (str): Earth Engine project ID
- `max_retries` (int): Maximum authentication attempts. Default: 3

**Returns**: bool - True if successful, False otherwise

**Behavior**:
1. Attempts to initialize with provided project ID
2. On failure, attempts authentication
3. Retries with exponential backoff
4. Logs all attempts and errors

**Example**:
```python
from landsat_downloader import initialize_earth_engine

if initialize_earth_engine('circumarcticzerocurtain'):
    print("Earth Engine initialized successfully")
else:
    print("Failed to initialize Earth Engine")
    sys.exit(1)
```

---

## Processing Functions

### Region Processing

#### `create_complete_coverage_regions()`
```python
def create_complete_coverage_regions() -> List[Dict]
```

Generate comprehensive region grid ensuring complete circumarctic coverage.

**Parameters**: None (uses Config class values)

**Returns**: List[Dict] - List of region definitions:
```python
[
    {
        'name': str,           # e.g., 'region_-180_45'
        'coords': List[float]  # [lon_min, lat_min, lon_max, lat_max]
    },
    ...
]
```

**Coverage Strategy**:
1. Main grid: LON_STEP × LAT_STEP cells covering entire domain
2. Dateline regions: Special handling for ±180° longitude
3. Polar cap regions: Finer resolution for 85-90°N

**Example**:
```python
from landsat_downloader import create_complete_coverage_regions

regions = create_complete_coverage_regions()
print(f"Total regions: {len(regions)}")

# Find polar regions
polar = [r for r in regions if 'pole' in r['name']]
print(f"Polar cap regions: {len(polar)}")
```

#### `process_region()`
```python
def process_region(
    region_info: Dict,
    queue: multiprocessing.Queue,
    stats_dict: dict,
    stats_lock: multiprocessing.Lock,
    start_date: str,
    end_date: str,
    cloud_threshold: int,
    output_dirs: Dict[str, Path]
) -> None
```

Process a single region: query Earth Engine, process scenes, save results.

**Parameters**:
- `region_info` (Dict): Region definition from `create_complete_coverage_regions()`
- `queue` (Queue): Multiprocessing queue for inter-process communication
- `stats_dict` (dict): Shared dictionary for statistics
- `stats_lock` (Lock): Lock for thread-safe statistics updates
- `start_date` (str): Start date in YYYY-MM-DD format
- `end_date` (str): End date in YYYY-MM-DD format
- `cloud_threshold` (int): Maximum cloud cover percentage
- `output_dirs` (Dict[str, Path]): Output directory structure

**Returns**: None (results communicated via queue)

**Process Flow**:
1. Check for existing checkpoint
2. Initialize Earth Engine
3. Query Landsat collection for region
4. Process scenes in batches
5. Save results to CSV
6. Create checkpoint
7. Update shared statistics

**Example**:
```python
# Typically called from multiprocessing context
import multiprocessing
from landsat_downloader import process_region

region = {
    'name': 'region_-180_45',
    'coords': [-180, 45, -170, 50]
}

manager = multiprocessing.Manager()
queue = manager.Queue()
stats_dict = manager.dict()
stats_lock = manager.Lock()

process_region(
    region, queue, stats_dict, stats_lock,
    '2020-01-01', '2023-12-31', 10,
    output_dirs
)
```

### Scene Processing

#### `get_scene_sample_points()`
```python
def get_scene_sample_points(
    scene: ee.Image,
    grid_size: int = 10
) -> Optional[ee.FeatureCollection]
```

Create systematic grid of sample points for a Landsat scene.

**Parameters**:
- `scene` (ee.Image): Earth Engine Image object
- `grid_size` (int): Number of points per side. Default: 10

**Returns**: Optional[ee.FeatureCollection] - Collection of sample points, or None on error

**Algorithm**:
1. Get scene footprint bounds
2. Apply 5% edge buffer (inset)
3. Create regular grid of points
4. Return as FeatureCollection

**Example**:
```python
import ee
from landsat_downloader import get_scene_sample_points

# Get a Landsat scene
scene = ee.Image('LANDSAT/LC08/C02/T1_L2/LC08_001001_20200101')

# Generate sample points
points = get_scene_sample_points(scene, grid_size=10)

if points:
    print(f"Created {grid_size * grid_size} sample points")
```

#### `process_single_scene()`
```python
@with_retry()
def process_single_scene(scene: ee.Image) -> List[Dict]
```

Process single Landsat scene to extract band values at sample points.

**Parameters**:
- `scene` (ee.Image): Earth Engine Image to process

**Returns**: List[Dict] - List of extracted data points:
```python
[
    {
        'scene_id': str,
        'acquisition_date': str,  # YYYY-MM-DD
        'longitude': float,
        'latitude': float,
        'B2': int,  # Blue band (scaled)
        'B3': int,  # Green band (scaled)
        'B4': int,  # Red band (scaled)
        'B10': int, # Thermal band (scaled)
        'cloud_cover': float
    },
    ...
]
```

**Decorator**: `@with_retry()` - Automatic retry with exponential backoff

**Example**:
```python
from landsat_downloader import process_single_scene
import ee

scene = ee.Image('LANDSAT/LC08/C02/T1_L2/LC08_001001_20200101')
results = process_single_scene(scene)

print(f"Extracted {len(results)} points from scene")
for point in results[:3]:
    print(f"  Lon: {point['longitude']:.4f}, Lat: {point['latitude']:.4f}")
```

#### `process_scenes_batch()`
```python
@with_retry()
def process_scenes_batch(
    scenes: List[ee.Image],
    region_name: str
) -> List[Dict]
```

Process batch of scenes in parallel.

**Parameters**:
- `scenes` (List[ee.Image]): List of Earth Engine Images
- `region_name` (str): Region identifier for logging

**Returns**: List[Dict] - Combined results from all scenes

**Parallelization**: Uses ThreadPoolExecutor with MAX_THREADS_PER_PROCESS workers

**Example**:
```python
from landsat_downloader import process_scenes_batch
import ee

collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
    .filterDate('2020-01-01', '2020-12-31') \
    .filterBounds(ee.Geometry.Rectangle([-180, 45, -170, 50])) \
    .limit(10)

scenes = [ee.Image(collection.toList(10).get(i)) for i in range(10)]
results = process_scenes_batch(scenes, 'test_region')

print(f"Total points: {len(results)}")
```

### Data Merging

#### `merger_process()`
```python
def merger_process(
    queue: multiprocessing.Queue,
    stats_dict: dict,
    stats_lock: multiprocessing.Lock,
    output_file: Path
) -> None
```

Merge CSV files into consolidated parquet file (runs as separate process).

**Parameters**:
- `queue` (Queue): Queue receiving file paths and status updates
- `stats_dict` (dict): Shared statistics dictionary
- `stats_lock` (Lock): Thread-safe lock for statistics
- `output_file` (Path): Final output parquet file path

**Queue Messages**:
```python
("data", file_path, point_count)     # Process data file
("complete", region_name, points)    # Region completed
("DONE", "", 0)                      # Terminate merger
```

**Process Flow**:
1. Monitor queue for incoming data files
2. Read CSV files as they arrive
3. Append to parquet file
4. Update shared statistics
5. Generate final summary on completion

**Example**:
```python
# Typically started as separate process in main()
import multiprocessing
from landsat_downloader import merger_process

manager = multiprocessing.Manager()
queue = manager.Queue()
stats_dict = manager.dict()
stats_lock = manager.Lock()
output_file = Path('raw/landsat_arctic_data.parquet')

merger = multiprocessing.Process(
    target=merger_process,
    args=(queue, stats_dict, stats_lock, output_file)
)
merger.start()

# ... processing regions ...

queue.put(("DONE", "", 0))
merger.join()
```

---

## Utility Functions

### Checkpoint Management

#### `load_completed_regions()`
```python
def load_completed_regions(checkpoint_dir: Path) -> set
```

Load set of completed region names from checkpoint file.

**Parameters**:
- `checkpoint_dir` (Path): Directory containing checkpoints

**Returns**: set - Set of completed region names

**Checkpoint File**: `completed_regions.json`
```json
{
    "completed_regions": ["region_-180_45", "region_-170_45", ...]
}
```

**Example**:
```python
from pathlib import Path
from landsat_downloader import load_completed_regions

checkpoint_dir = Path('checkpoints')
completed = load_completed_regions(checkpoint_dir)

print(f"Previously completed: {len(completed)} regions")
```

#### `save_completed_region()`
```python
def save_completed_region(
    region_name: str,
    checkpoint_dir: Path,
    stats_dict: Optional[dict] = None,
    stats_lock: Optional[multiprocessing.Lock] = None
) -> None
```

Mark region as completed in checkpoint file.

**Parameters**:
- `region_name` (str): Name of completed region
- `checkpoint_dir` (Path): Checkpoint directory
- `stats_dict` (Optional[dict]): Shared statistics dictionary
- `stats_lock` (Optional[Lock]): Lock for thread-safe updates

**Returns**: None

**Side Effects**:
- Updates `completed_regions.json`
- Updates shared statistics dictionary if provided

**Example**:
```python
from landsat_downloader import save_completed_region
from pathlib import Path

save_completed_region(
    'region_-180_45',
    Path('checkpoints')
)
```

### Logging

#### `setup_logging()`
```python
def setup_logging(log_dir: Path, debug: bool = False) -> logging.Logger
```

Configure logging with file and console handlers.

**Parameters**:
- `log_dir` (Path): Directory for log files
- `debug` (bool): Enable debug level logging. Default: False

**Returns**: logging.Logger - Configured logger instance

**Log File**: `landsat_download_YYYYMMDD_HHMMSS.log`

**Configuration**:
- File handler: DEBUG or INFO level
- Console handler: INFO level
- Format: `%(asctime)s [%(levelname)s] %(message)s`

**Example**:
```python
from pathlib import Path
from landsat_downloader import setup_logging

logger = setup_logging(Path('logs'), debug=True)
logger.info("Processing started")
logger.debug("Debug information")
```

### Decorators

#### `@with_retry()`
```python
def with_retry(
    max_retries: int = Config.RETRY_LIMIT,
    initial_delay: int = Config.RETRY_DELAY
) -> Callable
```

Exponential backoff retry decorator for functions.

**Parameters**:
- `max_retries` (int): Maximum retry attempts. Default: 10
- `initial_delay` (int): Initial delay in seconds. Default: 8

**Returns**: Callable - Decorated function

**Retry Logic**:
```python
delay = initial_delay * (2 ** attempt)
```

**Example**:
```python
from landsat_downloader import with_retry

@with_retry(max_retries=5, initial_delay=2)
def unstable_function():
    # Function that might fail
    result = risky_operation()
    return result

# Automatically retries up to 5 times with exponential backoff
result = unstable_function()
```

---

## Data Structures

### Region Definition
```python
{
    'name': str,           # Unique region identifier (e.g., 'region_-180_45')
    'coords': List[float], # [lon_min, lat_min, lon_max, lat_max]
    'type': str           # Optional: 'empty_cell', 'boundary', 'swath_gap'
}
```

### Coverage Analysis
```python
{
    'coverage_percentage': float,      # 0-100
    'total_cells': int,                # Total grid cells
    'filled_cells': int,               # Cells with data
    'empty_cells': int,                # Cells without data
    'gap_cells': List[Dict],           # List of gap cell definitions
    'boundary_issues': {
        'western_missing': bool,
        'eastern_missing': bool,
        'southern_missing': bool,
        'northern_missing': bool
    },
    'lon_range': [float, float],       # [min, max]
    'lat_range': [float, float]        # [min, max]
}
```

### Data Point
```python
{
    'scene_id': str,           # Landsat scene identifier
    'acquisition_date': str,   # YYYY-MM-DD format
    'longitude': float,        # Decimal degrees
    'latitude': float,         # Decimal degrees
    'B2': int,                 # Blue band (scaled integer)
    'B3': int,                 # Green band (scaled integer)
    'B4': int,                 # Red band (scaled integer)
    'B10': int,                # Thermal band (scaled integer)
    'cloud_cover': float       # Percentage (0-100)
}
```

### Processing Summary
```python
{
    'total_points': int,
    'unique_scenes': int,
    'date_range': [str, str],          # [min_date, max_date]
    'spatial_extent': {
        'lon_min': float,
        'lon_max': float,
        'lat_min': float,
        'lat_max': float
    },
    'regions_processed': int,
    'cloud_cover_range': [float, float] # [min, max]
}
```

---

## Band Scaling

### Surface Reflectance (B2, B3, B4)
```python
def scale_reflectance(band_value: int) -> float:
    """Convert scaled integer to reflectance (0-1)"""
    return band_value * 0.0000275 - 0.2
```

**Example**:
```python
b4_scaled = 8000
b4_reflectance = b4_scaled * 0.0000275 - 0.2
print(f"Red reflectance: {b4_reflectance:.4f}")
```

### Surface Temperature (B10)
```python
def scale_temperature(band_value: int) -> float:
    """Convert scaled integer to temperature (Kelvin)"""
    return band_value * 0.00341802 + 149.0
```

**Example**:
```python
b10_scaled = 30000
b10_kelvin = b10_scaled * 0.00341802 + 149.0
b10_celsius = b10_kelvin - 273.15
print(f"Temperature: {b10_celsius:.2f}°C")
```

---

## Error Handling

### Common Exceptions

#### Earth Engine Errors
```python
try:
    ee.Initialize(project='your-project')
except ee.EEException as e:
    logger.error(f"Earth Engine error: {e}")
    # Handle authentication or API errors
```

#### Processing Errors
```python
try:
    results = process_single_scene(scene)
except Exception as e:
    logger.error(f"Scene processing error: {e}")
    # Automatic retry via @with_retry decorator
```

#### File I/O Errors
```python
try:
    df = pd.read_parquet(file_path)
except FileNotFoundError:
    logger.error(f"File not found: {file_path}")
except pd.errors.ParserError as e:
    logger.error(f"Parse error: {e}")
```

---

## Performance Considerations

### Memory Management
```python
# Manual garbage collection after large operations
import gc

df = pd.read_parquet(large_file)
# ... process data ...
del df
gc.collect()
```

### Batch Processing
```python
# Process data in chunks to avoid memory issues
chunk_size = 10000
for start in range(0, len(data), chunk_size):
    end = min(start + chunk_size, len(data))
    chunk = data[start:end]
    process_chunk(chunk)
```

### Parallel Processing
```python
# Adjust parallelism based on resources
Config.MAX_CONCURRENT_PROCESSES = 4  # Reduce for limited systems
Config.MAX_THREADS_PER_PROCESS = 4
```

---

## Testing

### Unit Tests
```python
# Test sample point generation
def test_sample_points():
    scene = ee.Image('LANDSAT/LC08/C02/T1_L2/LC08_001001_20200101')
    points = get_scene_sample_points(scene, grid_size=4)
    assert points is not None
    assert points.size().getInfo() == 16  # 4x4 grid
```

### Integration Tests
```python
# Test full processing pipeline
def test_pipeline():
    # Create test region
    test_region = {
        'name': 'test_region',
        'coords': [-180, 45, -170, 50]
    }
    
    # Process region
    process_region(test_region, ...)
    
    # Verify output
    assert output_file.exists()
```

---

## Version History

| Version | Date       | Changes                                      |
|---------|------------|----------------------------------------------|
| 1.0.0   | 2025-01-20 | Initial API documentation                    |

---

## See Also

- [README.md](README.md) - Overview and quick start
- [PROCESSING_GUIDE.md](PROCESSING_GUIDE.md) - Detailed processing instructions
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues and solutions