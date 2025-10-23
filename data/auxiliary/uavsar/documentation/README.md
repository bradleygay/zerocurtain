#  UAVSAR Arctic Remote Sensing Pipeline

**Complete Alaska UAVSAR Dataset with SMAP Integration**


---

##  Complete Dataset

### **320 Million UAVSAR Observations**
- 8,197 displacement GeoTIFF files processed
- 30m spatial resolution
- Multiple Alaska sites (2012-2022)
- 32 GB compressed parquet

### **2.5 Million SMAP Observations**
- Alaska region (60-63°N, -167 to -163°W)
- Soil temperature and moisture
- 2015-2024 temporal coverage

### **Combined Dataset: 322.5 Million Records (34 GB)**

---

##  Quick Start

** Important:** These are large datasets. Always use column/row filtering!
```python
import pandas as pd

# Load ONLY the columns you need
df = pd.read_parquet('data_products/remote_sensing.parquet',
                      columns=['datetime', 'latitude', 'longitude', 'source'])

# Or filter by source
uavsar = pd.read_parquet('data_products/remote_sensing.parquet',
                          filters=[('source', '=', 'UAVSAR')])
```

See `DATASET_STATISTICS.md` for memory-efficient usage patterns.

---

##  File Locations

- **nisar.parquet:** `data_products/nisar.parquet` (32 GB)
- **remote_sensing.parquet:** `data_products/remote_sensing.parquet` (34 GB)
- **GeoTIFFs:** `displacement_30m/` (8,197 files)
- **Scripts:** `production/`
- **Documentation:** `documentation/`

---

See `DATASET_STATISTICS.md` and `USAGE_GUIDE.md` for complete documentation.
