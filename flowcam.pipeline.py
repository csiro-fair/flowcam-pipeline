"""Marimba Pipeline for the CSIRO FlowCam 8000."""  # noqa: N999

import re
import uuid
from datetime import datetime
from pathlib import Path
from shutil import copy2
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pytz
import typer
from ifdo.models import (
    ImageAcquisition,
    ImageData,
    ImageDeployment,
    ImageFaunaAttraction,
    ImageIllumination,
    ImageMarineZone,
    ImageNavigation,
    ImagePI,
    ImagePixelMagnitude,
    ImageQuality,
    ImageSpectralResolution,
)
from rich import print

from marimba.core.pipeline import BasePipeline
from marimba.core.utils.rich import error_panel
from marimba.core.wrappers.dataset import DatasetWrapper
from marimba.main import __version__


class FlowCamPipeline(BasePipeline):
    """
    Marimba Pipeline implementation for processing CSIRO FlowCam 8000.
    """

    MIN_REGION_AREA = 100  # Minimum area in pixels to consider a valid region
    MAX_PIXEL_STD_DEV = 80  # Maximum standard deviation of pixel intensities

    @staticmethod
    def get_pipeline_config_schema() -> dict[str, str]:
        """
        Get the pipeline configuration schema for the PLAOS pipeline.

        Returns:
            dict: Configuration parameters for the pipeline
        """
        return {
            "project_pi": "Joanna Strzelecki",
            "data_collector": "Joanna Strzelecki",
            "platform_id": "FC8000",
        }

    @staticmethod
    def get_collection_config_schema() -> dict[str, str]:
        """
        Get the collection configuration schema for the PLAOS pipeline.

        Returns:
            dict: Configuration parameters for the collection
        """
        return {
            "site_id": "CS6",
            "field_of_view": "1000",
        }

    @staticmethod
    def natural_sort_key(s: str) -> list[int | str]:
        """
        Extracts integers from a string and returns a tuple that can be used for sorting.
        """
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]

    def find_sub_images(self, img_path: Path, images_dir: Path, start_i: int) -> int:
        """
        Find and extract sub-images from a larger image based on contours.

        This function processes an input image to identify distinct regions (contours) and extracts them as separate
        sub-images. It uses OpenCV for image processing and contour detection. The function saves qualifying sub-images
        to the specified directory with sequentially numbered filenames.

        Args:
            img_path (Path): Path to the input image file.
            images_dir (Path): Directory where extracted sub-images will be saved.
            start_i (int): Starting index for naming the extracted sub-images.

        Returns:
            int: The next available index for naming subsequent sub-images.

        Raises:
            FileNotFoundError: If the input image file does not exist.
            PermissionError: If the function lacks permission to read the input file or write to the output directory.
            cv2.error: If there's an error in OpenCV operations (e.g., invalid image format).
        """
        # Load image and convert to grayscale
        img = cv2.imread(str(img_path))  # Convert Path object to string for cv2.imread
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Threshold image (black and white only)
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        # Find contours in the binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours by their position (top to bottom, left to right)
        contours = sorted(
            contours,
            key=lambda ctr: (cv2.boundingRect(ctr)[1], cv2.boundingRect(ctr)[0]),
        )

        # Get the base filename without the extension
        base_filename = img_path.stem.split("_")[0]

        i = start_i
        for contour in contours:
            # Get the rectangle that contains the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Ignore small regions
            if w * h > self.MIN_REGION_AREA:
                # Extract sub-image
                sub_img = img[y: y + h, x: x + w]

                # Check if the standard deviation of pixel intensities is below a certain threshold
                sub_img_float = sub_img.astype(float)
                if np.std(sub_img_float) < self.MAX_PIXEL_STD_DEV:
                    # Save the sub-image
                    output_path = images_dir / f"{base_filename}_{str(i).zfill(6)}.jpg"
                    cv2.imwrite(str(output_path), sub_img)
                    i += 1

        return i

    def _get_creation_date(self, data_dir: Path, site_id: str) -> datetime:
        """Extract creation date from station data."""
        match = re.search(r"\d+([A-Za-z]+)(\d+)", data_dir.parent.name)
        if not match:
            raise ValueError("Could not extract month and year from directory name")

        month, year = match.groups()
        station_data_df = pd.read_csv("/datasets/work/ev-flowcam-ml/source/station_data.csv")

        # Convert month name to number and prepare date filters
        month_num = pd.to_datetime(month, format="%B").month
        year_num = int(year)

        # Process station data
        station_data_df["date"] = pd.to_datetime(
            station_data_df["date"],
            format="%d/%m/%Y",
            errors="coerce",
        )

        station_row = station_data_df[
            (station_data_df["station_id"] == site_id)
            & (station_data_df["date"].dt.month == month_num)
            & (station_data_df["date"].dt.year == year_num)
            ]

        if len(station_row) == 0:
            raise ValueError("No matching data found")
        if len(station_row) > 1:
            raise ValueError("Multiple matching rows found")

        start_time_local = station_row.iloc[0]["start_time_local"]
        return datetime.strptime(start_time_local + " +0800", "%d/%m/%Y %H:%M %z")

    def _get_replicate_ids(self, source_path: Path) -> set[int]:
        """Extract unique replicate IDs from filenames in source directory."""
        rep_pattern = re.compile(r"rep(\d+)")
        rep_ids = set()

        for file in source_path.iterdir():
            if file.is_file() and (match := rep_pattern.search(file.name)):
                rep_ids.add(int(match.group(1)))

        return rep_ids

    def _process_replicate(
            self,
            rep_id: int,
            source_path: Path,
            data_dir: Path,
            config: dict[str, Any],
            date_dir: str,
    ) -> None:
        """Process all files for a single replicate."""
        # Create replicate directory
        rep_dir = data_dir / config["site_id"] / date_dir / str(rep_id).zfill(2)
        rep_dir.mkdir(exist_ok=True, parents=True)

        # Get sorted files for this replicate
        sorted_files = sorted(
            (f for f in source_path.iterdir() if f.is_file() and f"rep{rep_id}" in f.name),
            key=lambda p: self.natural_sort_key(p.name),
        )

        # Process each file based on its extension
        vignette_counter = 1
        for file_path in sorted_files:
            vignette_counter = self._process_file(
                file_path=file_path,
                rep_dir=rep_dir,
                vignette_counter=vignette_counter,
            )

    def _process_file(
            self,
            file_path: Path,
            rep_dir: Path,
            vignette_counter: int,
    ) -> int:
        """Process a single file based on its extension."""
        suffix = file_path.suffix.lower()

        if suffix == ".png":
            self.logger.info(f"Extracting vignettes from {file_path.name}")
            images_dir = rep_dir / "images"
            images_dir.mkdir(exist_ok=True, parents=True)
            return self.find_sub_images(file_path, images_dir, vignette_counter)

        if suffix == ".tif":
            cal_dir = rep_dir / "cal"
            cal_dir.mkdir(exist_ok=True, parents=True)
            copy2(file_path, cal_dir)

        elif suffix in [".csv", ".pdf"]:
            if not self.dry_run:
                rep_data_dir = rep_dir / "data"
                rep_data_dir.mkdir(exist_ok=True, parents=True)
                copy2(file_path, rep_data_dir)
            self.logger.debug(f"Copied {file_path.resolve().absolute()} -> {rep_dir}")

        return vignette_counter

    def _handle_error(self, message: str) -> None:
        """Handle errors uniformly throughout the importer."""
        self.logger.exception(message)
        print(error_panel(message))
        raise typer.Exit from None

    def _import(
            self,
            data_dir: Path,
            source_path: Path,
            config: dict[str, Any],
            **kwargs: dict[str, Any],  # noqa: ARG002
    ) -> None:
        """Import and process data from source path to data directory."""
        self.logger.info(f"Importing data from {source_path=} to {data_dir}")

        if not source_path.is_dir():
            return

        # Extract date information and get creation date
        creation_date = self._get_creation_date(data_dir, config["site_id"])
        date_dir = creation_date.strftime("%Y-%m-%d")

        # Process files for each replicate
        rep_ids = self._get_replicate_ids(source_path)
        if not rep_ids:
            print("No Rep ID found in the source path.")
            return

        for rep_id in rep_ids:
            self._process_replicate(
                rep_id=rep_id,
                source_path=source_path,
                data_dir=data_dir,
                config=config,
                date_dir=date_dir,
            )

    def _get_timestamp(self, data_dir: Path, site_id: str) -> str:
        """Extract timestamp from station data."""
        match = re.search(r"\d+([A-Za-z]+)(\d+)", data_dir.parent.name)
        if not match:
            raise ValueError("Could not extract month and year from directory name")

        month, year = match.groups()
        station_data_df = pd.read_csv("/datasets/work/ev-flowcam-ml/source/station_data.csv")

        month_num = pd.to_datetime(month, format="%B").month
        year_num = int(year)
        station_data_df["date"] = pd.to_datetime(
            station_data_df["date"],
            format="%d/%m/%Y",
            errors="coerce",
        )

        station_row = station_data_df[
            (station_data_df["station_id"] == site_id)
            & (station_data_df["date"].dt.month == month_num)
            & (station_data_df["date"].dt.year == year_num)
            ]

        if len(station_row) == 0:
            raise ValueError("No matching data found")
        if len(station_row) > 1:
            raise ValueError("Multiple matching rows found")

        start_time_local = station_row.iloc[0]["start_time_local"]
        creation_date = datetime.strptime(start_time_local + " +0800", "%d/%m/%Y %H:%M %z")
        return creation_date.strftime("%Y%m%dT%H%M%S+0800")

    def _extract_magnification(self, file_path: Path) -> str:
        """Extract magnification value from summary CSV file."""
        with Path.open(file_path) as file:
            content = file.read()

        match = re.search(r"Magnification,\s*([\d.]+)", content)
        return match.group(1) if match else "Magnification value not found"

    def _verify_csv_files(self, data_csv: Path, summary_csv: Path, rep_data_dir: Path) -> None:
        """
        Verify that required CSV files exist.

        Args:
            data_csv: Path to the data CSV file
            summary_csv: Path to the summary CSV file
            rep_data_dir: Directory containing the CSV files

        Raises:
            FileNotFoundError: If either CSV file is missing
        """
        if not data_csv.exists() or not summary_csv.exists():
            raise FileNotFoundError(f"Required CSV files not found in {rep_data_dir}")

    def _process_rep_directory(
            self,
            rep_dir: Path,
            data_dir: Path,
            site_id: str,
            iso_timestamp: str,
            config: dict[str, Any],
    ) -> None:
        """Process a single replicate directory."""
        rep_data_dir = rep_dir / "data"
        image_dir = rep_dir / "images"
        rep_id = rep_dir.name

        try:
            # Convert rep_id to integer - could raise ValueError
            rep_num = int(rep_id)
            # Construct file paths - could raise TypeError if path components are invalid
            data_csv = rep_data_dir / f"{data_dir.parent.name}rep{rep_num}.csv"
            summary_csv = rep_data_dir / f"{data_dir.parent.name}rep{rep_num}_summary.csv"

            # Verify CSV files exist using the abstracted function
            self._verify_csv_files(data_csv, summary_csv, rep_data_dir)

            self.logger.info(f"Found data CSV file: {data_csv}")

        except (ValueError, TypeError) as e:
            self._handle_error(f"Invalid replicate ID format in {rep_id}: {e!s}")
        except FileNotFoundError as e:
            self._handle_error(str(e))

        # Process the CSV file and rename images
        objective_mag = self._extract_magnification(summary_csv)
        self._process_images(
            data_csv=data_csv,
            image_dir=image_dir,
            data_dir=data_dir,
            site_id=site_id,
            rep_id=rep_id,
            objective_mag=objective_mag,
            iso_timestamp=iso_timestamp,
            config=config,
        )

    def _process_images(
            self,
            data_csv: Path,
            image_dir: Path,
            data_dir: Path,
            site_id: str,
            rep_id: str,
            objective_mag: str,
            iso_timestamp: str,
            config: dict[str, Any],
    ) -> None:
        """Process and rename images based on CSV data."""
        data_df = pd.read_csv(data_csv)
        data_df["Filename"] = ""

        # Validate that self.config exists
        if self.config is None:
            raise ValueError("Pipeline configuration is missing")

        # Get platform_id from config and validate it
        platform_id = self.config.get("platform_id")
        if not isinstance(platform_id, str):
            raise TypeError("platform_id must be provided in the pipeline config and must be a string")

        # Get field_of_view from config and validate it
        field_of_view = config.get("field_of_view")
        if not isinstance(field_of_view, str):
            raise TypeError("field_of_view must be provided in the collection config and must be a string")

        for index, row in data_df.iterrows():
            capture_id = str(row["Capture ID"]).zfill(6)
            image_filename = f"{row['Name']}_{capture_id}.jpg"
            image_path = image_dir / image_filename

            output_filename = (
                f"{platform_id}_"
                f"{site_id}_"
                f"{rep_id}_"
                f"{objective_mag}X_"
                f"{field_of_view}FOV_"
                f"{iso_timestamp}_"
                f"{capture_id}"
                ".JPG"
            )
            output_path = image_dir / output_filename

            # Store the new filename in the DataFrame - using .at for single-value assignment as it's more efficient and
            # type-safe than .loc for this specific use case
            data_df.at[index, "Filename"] = str(Path(output_path).relative_to(data_dir))  # noqa: PD008

            if not self.dry_run and image_path.exists():
                Path.rename(image_path, output_path)

        # Save the modified DataFrame
        data_df.to_csv(data_csv, index=False)

    def _process(
            self,
            data_dir: Path,
            config: dict[str, Any],
            **kwargs: dict[str, Any],  # noqa: ARG002
    ) -> None:
        """Process FlowCam data directory structure."""
        # Get site_id from config and validate it
        site_id = config.get("site_id")
        if not isinstance(site_id, str):
            raise TypeError("site_id must be provided in the collection config and must be a string")

        iso_timestamp = self._get_timestamp(data_dir, site_id)

        for site_dir in data_dir.iterdir():
            if not site_dir.is_dir():
                continue

            for date_dir in site_dir.iterdir():
                if not date_dir.is_dir():
                    continue

                for rep_dir in date_dir.iterdir():
                    if not rep_dir.is_dir():
                        continue

                    self._process_rep_directory(
                        rep_dir=rep_dir,
                        data_dir=data_dir,
                        site_id=site_dir.name,
                        iso_timestamp=iso_timestamp,
                        config=config,
                    )

    def _get_station_metadata(self, data_dir: Path, site_id: str) -> tuple[str, float, float, float]:
        """Extract station metadata from the CSV file."""
        match = re.search(r"\d+([A-Za-z]+)(\d+)", data_dir.parent.name)
        if not match:
            raise ValueError("Could not extract month and year from directory name")

        month, year = match.groups()
        station_data_df = pd.read_csv("/datasets/work/ev-flowcam-ml/source/station_data.csv")

        month_num = pd.to_datetime(month, format="%B").month
        year_num = int(year)
        station_data_df["date"] = pd.to_datetime(station_data_df["date"], format="%d/%m/%Y", errors="coerce")
        station_row = station_data_df[
            (station_data_df["station_id"] == site_id)
            & (station_data_df["date"].dt.month == month_num)
            & (station_data_df["date"].dt.year == year_num)
            ]

        if len(station_row) == 0:
            error_message = "No matching data found for the given site_id, month, and year."
            self.logger.exception(error_message)
            print(error_panel(error_message))
            raise typer.Exit from None
        if len(station_row) > 1:
            error_message = "Multiple matching rows found. Please check the data."
            self.logger.exception(error_message)
            print(error_panel(error_message))
            raise typer.Exit from None

        # Parse the timestamp and attach Perth timezone
        perth_tz = pytz.timezone("Australia/Perth")
        creation_date = datetime.strptime(
            station_row.iloc[0]["start_time_local"],
            "%d/%m/%Y %H:%M",
        ).replace(tzinfo=perth_tz)

        return (
            creation_date.strftime("%Y-%m-%d %H:%M:%S+0800"),
            station_row.iloc[0]["lat"],
            station_row.iloc[0]["lon"],
            -float(station_row.iloc[0]["max_net_depth_m"]),
        )

    def _create_image_data(
            self,
            row: dict[str, Any],
            image_datetime: str,
            latitude: float,
            longitude: float,
            depth: float,
    ) -> ImageData:
        """Create an ImageData object for a given row."""
        image_pi = ImagePI(name="Joanna Strzelecki", orcid="0000-0003-1138-2932")
        image_creators = [
            image_pi,
            ImagePI(name="Chris Jackett", orcid="0000-0003-1132-1558"),
        ]

        # Validate that self.config exists
        if self.config is None:
            raise ValueError("Pipeline configuration is missing")

        # Get platform_id from config and validate it
        platform_id = self.config.get("platform_id")
        if not isinstance(platform_id, str):
            raise TypeError("platform_id must be provided in the pipeline config and must be a string")

        # ruff: noqa: ERA001
        return ImageData(
            # iFDO core (required)
            image_datetime=image_datetime,
            image_latitude=latitude,
            image_longitude=longitude,
            image_altitude=depth,
            image_coordinate_reference_system="EPSG:4326",
            # image_coordinate_uncertainty_meters=None,
            # image_context=row["image_context"],
            # image_project=row["survey_id"],
            # image_event=f'{row["survey_id"]}_{row["deployment_number"]}',
            image_platform=platform_id,
            # image_sensor=str(row["camera_name"]).strip(),
            image_uuid=str(uuid.UUID(row["UUID"])),
            # image_hash_sha256=image_hash_sha256,
            image_pi=image_pi,
            image_creators=image_creators,
            image_license="CC BY-NC 4.0",
            image_copyright="CSIRO",
            # image_abstract=row["abstract"],
            #
            # # iFDO capture (optional)
            image_acquisition=ImageAcquisition.PHOTO,
            image_quality=ImageQuality.PRODUCT,
            image_deployment=ImageDeployment.SURVEY,
            image_navigation=ImageNavigation.SATELLITE,
            # image_scale_reference=ImageScaleReference.NONE,
            image_illumination=ImageIllumination.ARTIFICIAL_LIGHT,
            image_pixel_mag=ImagePixelMagnitude.UM,
            image_marine_zone=ImageMarineZone.WATER_COLUMN,
            image_spectral_resolution=ImageSpectralResolution.RGB,
            # image_capture_mode=ImageCaptureMode.MANUAL,
            image_fauna_attraction=ImageFaunaAttraction.NONE,
            # image_area_square_meter: Optional[float] = None
            # image_meters_above_ground: Optional[float] = None
            # image_acquisition_settings: Optional[dict] = None
            # image_camera_yaw_degrees: Optional[float] = None
            # image_camera_pitch_degrees: Optional[float] = None
            # image_camera_roll_degrees: Optional[float] = None
            # image_overlap_fraction=0,
            image_datetime_format="%Y-%m-%d %H:%M:%S±HHMM",
            # image_camera_pose: Optional[CameraPose] = None
            # image_camera_housing_viewport=camera_housing_viewport,
            # image_flatport_parameters: Optional[FlatportParameters] = None
            # image_domeport_parameters: Optional[DomeportParameters] = None
            # image_camera_calibration_model: Optional[CameraCalibrationModel] = None
            # image_photometric_calibration: Optional[PhotometricCalibration] = None
            # image_objective: Optional[str] = None
            image_target_environment="Pelagic",
            # image_target_timescale: Optional[str] = None
            # image_spatial_constraints: Optional[str] = None
            # image_temporal_constraints: Optional[str] = None
            # image_time_synchronization: Optional[str] = None
            image_item_identification_scheme="<platform_id>_<site_id>_<replicate_id>_<objective_magnification>_<field_of_view>_<datetimestamp>_<capture_id>.<ext>",
            image_curation_protocol=f"Processed with Marimba {__version__}",
            #
            # # iFDO content (optional)
            # image_entropy=0.0,
            # image_particle_count: Optional[int] = None
            # image_average_color=[0, 0, 0],
            # image_mpeg7_colorlayout: Optional[List[float]] = None
            # image_mpeg7_colorstatistics: Optional[List[float]] = None
            # image_mpeg7_colorstructure: Optional[List[float]] = None
            # image_mpeg7_dominantcolor: Optional[List[float]] = None
            # image_mpeg7_edgehistogram: Optional[List[float]] = None
            # image_mpeg7_homogenoustexture: Optional[List[float]] = None
            # image_mpeg7_stablecolor: Optional[List[float]] = None
            # image_annotation_labels: Optional[List[ImageAnnotationLabel]] = None
            # image_annotation_creators: Optional[List[ImageAnnotationCreator]] = None
            # image_annotations: Optional[List[ImageAnnotation]] = None
        )

    def _generate_summaries(
            self,
            data_mapping: dict[Path, tuple[Path, list[ImageData] | None, dict[str, Any] | None]],
            data_dir: Path,
    ) -> None:
        """Generate dataset summaries and iFDOs."""
        summary_directories: set[str] = set()
        ifdo_directories: set[str] = set()

        # Collect output directories
        for relative_dst, image_data_list, _ in data_mapping.values():
            if image_data_list:
                parts = relative_dst.parts
                if len(parts) > 0:
                    summary_directories.add(parts[0])
                    ifdo_directories.add(parts[0])
                if len(parts) > 1:
                    summary_directories.add(str(Path(parts[0]) / parts[1]))
                    ifdo_directories.add(str(Path(parts[0]) / parts[1]))

        # Convert the sets to sorted lists
        summary_dirs: list[str] = sorted(summary_directories)
        ifdo_dirs: list[str] = sorted(ifdo_directories)

        # Generate summaries
        for directory in summary_dirs:
            subset_data_mapping = {
                src.as_posix(): image_data_list
                for src, (relative_dst, image_data_list, _) in data_mapping.items()
                if str(relative_dst).startswith(directory) and image_data_list
            }

            # Create a dataset summary
            dataset_wrapper = DatasetWrapper(data_dir / directory, version=None, dry_run=True)
            dataset_wrapper.dry_run = False
            dataset_wrapper.generate_dataset_summary(subset_data_mapping, progress=False)

            # Add the summary to the dataset mapping
            output_file_path = dataset_wrapper.summary_path.relative_to(data_dir)
            data_mapping[dataset_wrapper.summary_path] = output_file_path, None, None

        # Generate iFDOs
        for directory in ifdo_dirs:
            subset_data_mapping = {
                relative_dst.relative_to(directory).as_posix(): image_data_list
                for src, (relative_dst, image_data_list, _) in data_mapping.items()
                if str(relative_dst).startswith(directory) and image_data_list
            }

            # Create an iFDO
            dataset_wrapper = DatasetWrapper(data_dir / directory, version=None, dry_run=True)
            dataset_wrapper.dry_run = False
            dataset_wrapper.generate_ifdo(directory, subset_data_mapping, progress=False)

            # Add the iFDO to the dataset mapping
            output_file_path = dataset_wrapper.metadata_path.relative_to(data_dir)
            data_mapping[dataset_wrapper.metadata_path] = output_file_path, None, None

    def _process_replicate_directory(
            self,
            rep_dir: Path,
            data_dir: Path,
            image_datetime: str,
            latitude: float,
            longitude: float,
            depth: float,
    ) -> dict[Path, tuple[Path, list[ImageData] | None, dict[str, Any] | None]]:
        """Process a single replicate directory and return its data mapping."""
        rep_data_dir = rep_dir / "data"
        rep_mapping: dict[Path, tuple[Path, list[ImageData] | None, dict[str, Any] | None]] = {}

        try:
            data_csv = rep_data_dir / f"{data_dir.parent.name}rep{int(rep_dir.name)}.csv"
            self.logger.info(f"Found data CSV file: {data_csv}")
        except Exception:
            error_message = f"No CSV files found in the {rep_data_dir!s} directory."
            self.logger.exception(error_message)
            print(error_panel(error_message))
            raise typer.Exit from None

        data_df = pd.read_csv(data_csv)

        for _index, row in data_df.iterrows():
            file_path = data_dir / row["Filename"]
            output_file_path = file_path.relative_to(data_dir)

            if file_path.is_file() and file_path.suffix.lower() in [".jpg"]:
                image_data_list = self._create_image_data(
                    row.to_dict(),
                    image_datetime,
                    latitude,
                    longitude,
                    depth,
                )
                rep_mapping[file_path] = output_file_path, [image_data_list], row.to_dict()

        return rep_mapping

    def _collect_ancillary_files(self, data_dir: Path) -> dict[Path, tuple[Path, None, None]]:
        """Collect all ancillary files that aren't in image directories."""
        ancillary_mapping = {}
        all_files = data_dir.glob("**/*")
        ancillary_files = [f for f in all_files if "images" not in f.parts]

        for file_path in ancillary_files:
            if file_path.is_file():
                output_file_path = file_path.relative_to(data_dir)
                ancillary_mapping[file_path] = output_file_path, None, None

        return ancillary_mapping

    def _package(
            self,
            data_dir: Path,
            config: dict[str, Any],
            **kwargs: dict[str, Any],  # noqa: ARG002
    ) -> dict[Path, tuple[Path, list[ImageData] | None, dict[str, Any] | None]]:
        """Package the data directory into a standardized format."""
        data_mapping: dict[Path, tuple[Path, list[ImageData] | None, dict[str, Any] | None]] = {}

        # Get site_id from config and validate it
        site_id = config.get("site_id")
        if not isinstance(site_id, str):
            raise TypeError("site_id must be provided in the collection config and must be a string")

        # Get station metadata
        image_datetime, latitude, longitude, depth = self._get_station_metadata(data_dir, site_id)

        # Add ancillary files to data mapping
        data_mapping.update(self._collect_ancillary_files(data_dir))

        # Process each replicate directory
        for site_id in data_dir.iterdir():
            if not site_id.is_dir():
                continue

            for date_dir in site_id.iterdir():
                if not date_dir.is_dir():
                    continue

                for rep_id in date_dir.iterdir():
                    if not rep_id.is_dir():
                        continue

                    rep_dir = data_dir / site_id.name / date_dir / rep_id.name
                    rep_mapping = self._process_replicate_directory(
                        rep_dir,
                        data_dir,
                        image_datetime,
                        latitude,
                        longitude,
                        depth,
                    )
                    data_mapping.update(rep_mapping)

        # Generate summaries and iFDOs
        self._generate_summaries(data_mapping, data_dir)

        return data_mapping
