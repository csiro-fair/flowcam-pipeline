# FlowCam Pipeline

A Marimba Pipeline for automated processing of plankton imagery collections from the FlowCam imaging flow cytometer. 
The Pipeline specializes in extracting individual vignettes from VisualSpreadsheet collage images while preserving 
comprehensive measurement metadata and spatio-temporal context.


## Overview

The FlowCam Pipeline is designed to process plankton imagery collected from marine monitoring campaigns. It handles 
data from the FlowCam 8400, which captures low-resolution greyscale images of isolated particles in suspension.

Key capabilities include:

- Automated extraction of individual vignettes from collage images using computer vision
- Quality-controlled particle detection and isolation
- Preservation of morphological and optical measurements
- Integration of station metadata including GPS coordinates, sampling depths, and timestamps
- Generation of FAIR-compliant datasets with embedded metadata


## Requirements

The FlowCam Pipeline is built on the [Marimba](https://github.com/csiro-fair/marimba) framework which includes all 
necessary dependencies for this Pipeline. No additional packages are required beyond those installed with Marimba.


## Installation

Create a new Marimba project and add the FlowCam Pipeline:

```bash
marimba new project my-flowcam-project
cd my-flowcam-project
marimba new pipeline my-flowcam-pipeline https://github.com/csiro-fair/flowcam-pipeline.git \
--config '{"platform_id": "FC8400"}'
```

## Configuration

### Pipeline Configuration
The Pipeline requires:
- `platform_id`: FlowCam instrument identifier (default: "FC8400")

### Collection Configuration
Each Collection requires:
- `site_id`: Sampling station identifier
- `field_of_view`: Flow cell field of view in microns (default: "1000")


## Usage

### Importing

Import collections with site-specific configurations:
```bash
marimba import collection-one '/path/to/source/station-one/' \
--config '{"site_id": "SITE01", "field_of_view": "1000"}'
```

For monitoring campaigns, multiple stations can be processed in batch:
```bash
# Import multiple stations
marimba import collection-one '/path/to/station-one/' --config '{"site_id": "SITE01", "field_of_view": "1000"}'
marimba import collection-two '/path/to/station-two/' --config '{"site_id": "SITE02", "field_of_view": "1000"}'
marimba import collection-three '/path/to/station-three/' --config '{"site_id": "SITE03", "field_of_view": "1000"}'
```

### Source Data Structure

The Pipeline expects FlowCam data containing sample replicates:
```
source/
└── station-one/
    ├── rep1_summary.csv      # Run summary metadata
    ├── rep1.csv              # Particle measurements
    ├── rep1.png              # Collage image
    ├── rep2_summary.csv
    ├── rep2.csv
    └── rep2.png
```

### Processing

```bash
marimba process
```

During processing, the FlowCam Pipeline:
1. Creates a hierarchical directory structure by station, date, and replicate
2. Extracts vignettes from collage images using computer vision
3. Integrates station metadata (coordinates, depth, timestamp)
4. Preserves morphological measurements and optical properties

### Packaging

```bash
marimba package my-flowcam-dataset \
--operation link \
--version 1.0 \
--contact-name "Keiko Abe" \
--contact-email "keiko.abe@email.com"
```

The `--operation link` flag creates hard links instead of copying files, optimizing storage for large datasets.


## Processed Data Structure

```                
PCW_2022/                                                                                           # Root dataset directory
├── data/                                                                                           # Directory containing all processed data
│   └── FC8400/                                                                                     # FlowCam instrument-specific data directory
│       └── [CS|OW]*/                                                                               # Station directories (CS17, OW43, etc.)
│           └── YYYY-MM-DD/                                                                         # Date-based directories
│               └── ##/                                                                             # Sequential run numbers (01, 02, etc.)
│                   ├── data/                                                                       # Run-specific data files
│                   │   ├── [Station][Month][Year]rep[#].csv                                        # Main data file
│                   │   └── [Station][Month][Year]rep[#]_summary.csv                                # Summary statistics
│                   └── images/                                                                     # Run-specific images
│                       └── FC8400_[Station]##_[Magnification]_[FOV]_[Timestamp]_[Capture_ID].JPG   # Image files
├── logs/                                                                                           # Directory containing all processing logs
│   ├── pipelines/                                                                                  # Pipeline-specific logs
│   │   └── FC8400.log                                                                              # Logs from FlowCam Pipeline
│   ├── dataset.log                                                                                 # Dataset packaging logs
│   └── project.log                                                                                 # Overall project processing logs
├── pipelines/                                                                                      # Directory containing pipeline code
│   └── FC8400/                                                                                     # Pipeline-specific directory
│       ├── repo/                                                                                   # Pipeline source code repository
│       │   ├── flowcam.pipeline.py                                                                 # Pipeline implementation
│       │   ├── LICENSE                                                                             # Pipeline license file
│       │   └── README.md                                                                           # Pipeline README file
│       └── pipeline.yml                                                                            # Pipeline configuration
├── ifdo.yml                                                                                        # Dataset-level iFDO metadata file
├── manifest.txt                                                                                    # File manifest with SHA256 hashes
├── map.png                                                                                         # Spatial visualization of dataset
└── summary.md                                                                                      # Dataset summary and statistics
```


## Metadata

The FlowCam Pipeline captures comprehensive metadata including:

### Station Metadata
- GPS coordinates
- Sampling depths
- Collection times
- Environmental parameters

### Technical Metadata
- Image acquisition parameters
- FlowCam configuration
- Processing parameters
- Quality metrics

### Image-Specific Measurements
- Morphological parameters (area, diameter, circularity)
- Optical properties (RGB intensities, transparency)
- Volumetric calculations (biovolume across geometric models)
- Positional data (capture coordinates, timestamps)

All metadata is standardized using the iFDO schema (v2.1.0) and embedded in both image EXIF tags and dataset-level files.


## Contributors

The FlowCam Pipeline was developed by:
- Christopher Jackett (CSIRO)
- Joanna Strzelecki (CSIRO)
- Ruth Eriksen (CSIRO)
- Julian Uribe-Palomino (CSIRO)
- James McLaughlin (CSIRO)


## License

The FlowCam Pipeline is distributed under the [CSIRO BSD/MIT](LICENSE) license.


## Contact

For inquiries related to this repository, please contact:

- **Chris Jackett**  
  *Software Engineer, CSIRO*  
  Email: [chris.jackett@csiro.au](mailto:chris.jackett@csiro.au)
