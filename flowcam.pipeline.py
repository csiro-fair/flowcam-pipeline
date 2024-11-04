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
    def get_pipeline_config_schema() -> dict:
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
    def get_collection_config_schema() -> dict:
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
                if np.std(sub_img) < self.MAX_PIXEL_STD_DEV:
                    # Save the sub-image
                    output_path = images_dir / f"{base_filename}_{str(i).zfill(6)}.jpg"
                    cv2.imwrite(str(output_path), sub_img)  # Convert Path object to string for cv2.imwrite
                    i += 1

        return i

    def _import(
            self,
            data_dir: Path,
            source_path: Path,
            config: dict[str, Any],
            **kwargs: dict,  # noqa: ARG002
    ) -> None:
        # This is really a process to extract the individual vignettes from the collage images
        self.logger.info(f"Importing data from {source_path=} to {data_dir}")

        if not source_path.is_dir():
            return

        month, year = re.search(r"\d+([A-Za-z]+)(\d+)", data_dir.parent.name).groups()
        station_data_df = pd.read_csv("/datasets/work/ev-flowcam-ml/source/station_data.csv")
        site_id = config.get("site_id")

        # Get the row where station_id equals site_id
        month_num = pd.to_datetime(month, format="%B").month
        year_num = int(year)
        station_data_df["date"] = pd.to_datetime(station_data_df["date"], format="%d/%m/%Y", errors="coerce")
        station_row = station_data_df[
            (station_data_df["station_id"] == site_id)
            & (station_data_df["date"].dt.month == month_num)
            & (station_data_df["date"].dt.year == year_num)
            ]

        # Check if exactly one row is returned
        if len(station_row) == 1:
            start_time_local = station_row.iloc[0]["start_time_local"]
            creation_date = datetime.strptime(start_time_local + " +0800", "%d/%m/%Y %H:%M %z")
            # ISO 8601 Date Format
            date_dir = creation_date.strftime("%Y-%m-%d")
        elif len(station_row) == 0:
            error_message = "No matching data found for the given site_id, month, and year."
            self.logger.exception(error_message)
            print(error_panel(error_message))
            raise typer.Exit from None
        else:
            error_message = "Multiple matching rows found. Please check the data."
            self.logger.exception(error_message)
            print(error_panel(error_message))
            raise typer.Exit from None

        # Regular expression to match the rep number in the filenames
        rep_pattern = re.compile(r"rep(\d+)")

        # Find all files in the directory
        files = [f for f in source_path.iterdir() if f.is_file()]

        # Extract rep IDs
        rep_ids = set()
        for file in files:
            match = rep_pattern.search(file.name)
            if match:
                rep_ids.add(int(match.group(1)))  # Convert the matched group to integer
        if not rep_ids:
            print("No Rep ID found in the source path.")

        for rep_id in rep_ids:

            rep_dir = data_dir / config.get("site_id") / date_dir / str(rep_id).zfill(2)
            rep_dir.mkdir(exist_ok=True, parents=True)

            sorted_file_names = sorted(
                (f for f in source_path.iterdir() if f.is_file() and f"rep{rep_id}" in f.name),
                key=lambda p: self.natural_sort_key(p.name),
            )

            i = 1
            for file_path in sorted_file_names:
                if file_path.suffix.lower() == ".png":
                    self.logger.info(f"Extracting vignettes from {file_path.name}")
                    images_dir = rep_dir / "images"
                    images_dir.mkdir(exist_ok=True, parents=True)
                    i = self.find_sub_images(file_path, images_dir, i)
                if file_path.suffix.lower() == ".tif":
                    cal_dir = rep_dir / "cal"
                    cal_dir.mkdir(exist_ok=True, parents=True)
                    copy2(file_path, cal_dir)
                if file_path.suffix.lower() in [".csv", ".pdf"]:
                    if not self.dry_run:
                        rep_data_dir = rep_dir / "data"
                        rep_data_dir.mkdir(exist_ok=True, parents=True)
                        copy2(file_path, rep_data_dir)
                    self.logger.debug(f"Copied {file_path.resolve().absolute()} -> {rep_dir}")

    def _process(
            self,
            data_dir: Path,
            config: dict[str, Any],
            **kwargs: dict,  # noqa: ARG002
    ) -> None:

        month, year = re.search(r"\d+([A-Za-z]+)(\d+)", data_dir.parent.name).groups()
        station_data_df = pd.read_csv("/datasets/work/ev-flowcam-ml/source/station_data.csv")
        site_id = config.get("site_id")

        # Get the row where station_id equals site_id
        month_num = pd.to_datetime(month, format="%B").month
        year_num = int(year)
        station_data_df["date"] = pd.to_datetime(station_data_df["date"], format="%d/%m/%Y", errors="coerce")
        station_row = station_data_df[
            (station_data_df["station_id"] == site_id)
            & (station_data_df["date"].dt.month == month_num)
            & (station_data_df["date"].dt.year == year_num)
            ]

        # Check if exactly one row is returned
        if len(station_row) == 1:
            start_time_local = station_row.iloc[0]["start_time_local"]
            # Parse the timestamp with explicit UTC+8 timezone
            creation_date = datetime.strptime(start_time_local + " +0800", "%d/%m/%Y %H:%M %z")
            iso_timestamp = creation_date.strftime("%Y%m%dT%H%M%S+0800")
        elif len(station_row) == 0:
            error_message = "No matching data found for the given site_id, month, and year."
            self.logger.exception(error_message)
            print(error_panel(error_message))
            raise typer.Exit from None
        else:
            error_message = "Multiple matching rows found. Please check the data."
            self.logger.exception(error_message)
            print(error_panel(error_message))
            raise typer.Exit from None

        for site_id in data_dir.iterdir():
            if site_id.is_dir():
                for date_dir in site_id.iterdir():
                    if date_dir.is_dir():
                        for rep_id in date_dir.iterdir():
                            if rep_id.is_dir():
                                rep_dir = data_dir / site_id.name / date_dir / rep_id.name
                                rep_data_dir = rep_dir / "data"
                                image_dir = rep_dir / "images"

                                try:
                                    data_csv = rep_data_dir / f"{data_dir.parent.name}rep{int(rep_id.name)}.csv"
                                    summary_csv = rep_data_dir / f"{data_dir.parent.name}rep{int(rep_id.name)}_summary.csv"
                                    self.logger.info(f"Found data CSV file: {data_csv}")
                                except Exception:
                                    error_message = f"No CSV files found in the {rep_data_dir!s} directory."
                                    self.logger.exception(error_message)
                                    print(error_panel(error_message))
                                    raise typer.Exit from None

                                def extract_magnification(file_path: Path) -> str:
                                    with Path.open(file_path) as file:
                                        content = file.read()

                                    # Using regular expression to find 'Magnification: value'
                                    match = re.search(r"Magnification,\s*([\d.]+)", content)
                                    if match:
                                        return match.group(1)  # Return the value found after 'Magnification:'
                                    return "Magnification value not found"

                                data_df = pd.read_csv(data_csv)
                                data_df["Filename"] = ""
                                objective_mag = extract_magnification(summary_csv)

                                # Loop through each row in the DataFrame
                                for index, row in data_df.iterrows():
                                    capture_id = str(row["Capture ID"]).zfill(6)

                                    image_filename = f"{row['Name']}_{capture_id}.jpg"
                                    image_path = image_dir / image_filename

                                    output_filename = (
                                        f'{self.config.get("platform_id")}_'
                                        f"{site_id.name}_"
                                        f"{rep_id.name}_"
                                        f"{objective_mag}X_"
                                        f'{config.get("field_of_view")}FOV_'
                                        f"{iso_timestamp}_"
                                        f"{capture_id}"
                                        ".JPG"
                                    )
                                    output_path = image_dir / output_filename

                                    # Store the new filename in the DataFrame
                                    data_df.loc[index, "Filename"] = Path(output_path).relative_to(data_dir)

                                    if not self.dry_run and image_path.exists():
                                        Path.rename(image_path, output_path)

                                # Save the modified DataFrame
                                data_df.to_csv(
                                    rep_data_dir / f"{data_dir.parent.name}rep{int(rep_id.name)}.csv",
                                    index=False,
                                )

    # ruff: noqa: ERA001, E501
    # def _package(
    #         self,
    #         data_dir: Path,
    #         config: dict[str, Any],
    #         **kwargs: dict,
    # ) -> dict[Path, tuple[Path, list[ImageData] | None, dict[str, Any] | None]]:
    #     data_mapping: dict[Path, tuple[Path, list[ImageData] | None, dict[str, Any] | None]] = {}
    #
    #     month, year = re.search(r"\d+([A-Za-z]+)(\d+)", data_dir.parent.name).groups()
    #     station_data_df = pd.read_csv("/datasets/work/ev-flowcam-ml/source/station_data.csv")
    #     site_id = config.get("site_id")
    #
    #     # Get the row where station_id equals site_id
    #     month_num = pd.to_datetime(month, format="%B").month
    #     year_num = int(year)
    #     station_data_df["date"] = pd.to_datetime(station_data_df["date"], format="%d/%m/%Y", errors="coerce")
    #     station_row = station_data_df[
    #         (station_data_df["station_id"] == site_id)
    #         & (station_data_df["date"].dt.month == month_num)
    #         & (station_data_df["date"].dt.year == year_num)
    #         ]
    #
    #     # Check if exactly one row is returned
    #     if len(station_row) == 1:
    #         # Extract latitude and longitude
    #         latitude = station_row.iloc[0]["lat"]
    #         longitude = station_row.iloc[0]["lon"]
    #         depth = -float(station_row.iloc[0]["max_net_depth_m"])
    #         start_time_local = station_row.iloc[0]["start_time_local"]
    #
    #         # Parse the timestamp and attach Perth timezone
    #         perth_tz = pytz.timezone("Australia/Perth")
    #         creation_date = datetime.strptime(start_time_local, "%d/%m/%Y %H:%M").replace(tzinfo=perth_tz)
    #         image_datetime = creation_date.strftime("%Y-%m-%d %H:%M:%S+0800")
    #     elif len(station_row) == 0:
    #         error_message = "No matching data found for the given site_id, month, and year."
    #         self.logger.exception(error_message)
    #         print(error_panel(error_message))
    #         raise typer.Exit from None
    #     else:
    #         error_message = "Multiple matching rows found. Please check the data."
    #         self.logger.exception(error_message)
    #         print(error_panel(error_message))
    #         raise typer.Exit from None
    #
    #     for site_id in data_dir.iterdir():
    #         if site_id.is_dir():
    #             for date_dir in site_id.iterdir():
    #                 if date_dir.is_dir():
    #                     for rep_id in date_dir.iterdir():
    #                         if rep_id.is_dir():
    #                             rep_dir = data_dir / site_id.name / date_dir / rep_id.name
    #                             rep_data_dir = rep_dir / "data"
    #
    #                             all_files = data_dir.glob("**/*")
    #
    #                             # Filter out files that are in or under any 'images' directory
    #                             ancillary_files = [f for f in all_files if "images" not in f.parts]
    #
    #                             # Add ancillary files to data mapping
    #                             for file_path in ancillary_files:
    #                                 if file_path.is_file():
    #                                     output_file_path = file_path.relative_to(data_dir)
    #                                     data_mapping[file_path] = output_file_path, None, None
    #
    #                             try:
    #                                 data_csv = rep_data_dir / f"{data_dir.parent.name}rep{int(rep_id.name)}.csv"
    #                                 self.logger.info(f"Found data CSV file: {data_csv}")
    #                             except Exception:
    #                                 error_message = f"No CSV files found in the {rep_data_dir!s} directory."
    #                                 self.logger.exception(error_message)
    #                                 print(error_panel(error_message))
    #                                 raise typer.Exit from None
    #
    #                             data_df = pd.read_csv(data_csv)
    #
    #                             for _index, row in data_df.iterrows():
    #                                 file_path = data_dir / row["Filename"]
    #                                 output_file_path = file_path.relative_to(data_dir)
    #
    #                                 if file_path.is_file() and file_path.suffix.lower() in [".jpg"]:
    #                                     image_creators = [
    #                                         ImagePI(name="Joanna Strzelecki", orcid="0000-0003-1138-2932"),
    #                                         ImagePI(name="Chris Jackett", orcid="0000-0003-1132-1558"),
    #                                     ]
    #                                     image_pi = ImagePI(name="Joanna Strzelecki", orcid="0000-0003-1138-2932")
    #
    #                                     # ruff: noqa: ERA001
    #                                     image_data_list = ImageData(
    #                                         # iFDO core (required)
    #                                         image_datetime=image_datetime,
    #                                         image_latitude=latitude,
    #                                         image_longitude=longitude,
    #                                         image_altitude=depth,
    #                                         image_coordinate_reference_system="EPSG:4326",
    #                                         # image_coordinate_uncertainty_meters=None,
    #                                         # image_context=row["image_context"],
    #                                         # image_project=row["survey_id"],
    #                                         # image_event=f'{row["survey_id"]}_{row["deployment_number"]}',
    #                                         image_platform=self.config.get("platform_id"),
    #                                         # image_sensor=str(row["camera_name"]).strip(),
    #                                         image_uuid=str(uuid.UUID(row["UUID"])),
    #                                         # image_hash_sha256=image_hash_sha256,
    #                                         image_pi=image_pi,
    #                                         image_creators=image_creators,
    #                                         image_license="CC BY-NC 4.0",
    #                                         image_copyright="CSIRO",
    #                                         # image_abstract=row["abstract"],
    #                                         #
    #                                         # # iFDO capture (optional)
    #                                         image_acquisition=ImageAcquisition.PHOTO,
    #                                         image_quality=ImageQuality.PRODUCT,
    #                                         image_deployment=ImageDeployment.SURVEY,
    #                                         image_navigation=ImageNavigation.SATELLITE,
    #                                         # image_scale_reference=ImageScaleReference.NONE,
    #                                         image_illumination=ImageIllumination.ARTIFICIAL_LIGHT,
    #                                         image_pixel_mag=ImagePixelMagnitude.UM,
    #                                         image_marine_zone=ImageMarineZone.WATER_COLUMN,
    #                                         image_spectral_resolution=ImageSpectralResolution.RGB,
    #                                         # image_capture_mode=ImageCaptureMode.MANUAL,
    #                                         image_fauna_attraction=ImageFaunaAttraction.NONE,
    #                                         # image_area_square_meter: Optional[float] = None
    #                                         # image_meters_above_ground: Optional[float] = None
    #                                         # image_acquisition_settings: Optional[dict] = None
    #                                         # image_camera_yaw_degrees: Optional[float] = None
    #                                         # image_camera_pitch_degrees: Optional[float] = None
    #                                         # image_camera_roll_degrees: Optional[float] = None
    #                                         # image_overlap_fraction=0,
    #                                         image_datetime_format="%Y-%m-%d %H:%M:%S±HHMM",
    #                                         # image_camera_pose: Optional[CameraPose] = None
    #                                         # image_camera_housing_viewport=camera_housing_viewport,
    #                                         # image_flatport_parameters: Optional[FlatportParameters] = None
    #                                         # image_domeport_parameters: Optional[DomeportParameters] = None
    #                                         # image_camera_calibration_model: Optional[CameraCalibrationModel] = None
    #                                         # image_photometric_calibration: Optional[PhotometricCalibration] = None
    #                                         # image_objective: Optional[str] = None
    #                                         image_target_environment="Pelagic",
    #                                         # image_target_timescale: Optional[str] = None
    #                                         # image_spatial_constraints: Optional[str] = None
    #                                         # image_temporal_constraints: Optional[str] = None
    #                                         # image_time_synchronization: Optional[str] = None
    #                                         image_item_identification_scheme="<platform_id>_<site_id>_<replicate_id>_<objective_magnification>_<field_of_view>_<datetimestamp>_<capture_id>.<ext>",
    #                                         image_curation_protocol=f"Processed with Marimba {__version__}",
    #                                         #
    #                                         # # iFDO content (optional)
    #                                         # image_entropy=0.0,
    #                                         # image_particle_count: Optional[int] = None
    #                                         # image_average_color=[0, 0, 0],
    #                                         # image_mpeg7_colorlayout: Optional[List[float]] = None
    #                                         # image_mpeg7_colorstatistics: Optional[List[float]] = None
    #                                         # image_mpeg7_colorstructure: Optional[List[float]] = None
    #                                         # image_mpeg7_dominantcolor: Optional[List[float]] = None
    #                                         # image_mpeg7_edgehistogram: Optional[List[float]] = None
    #                                         # image_mpeg7_homogenoustexture: Optional[List[float]] = None
    #                                         # image_mpeg7_stablecolor: Optional[List[float]] = None
    #                                         # image_annotation_labels: Optional[List[ImageAnnotationLabel]] = None
    #                                         # image_annotation_creators: Optional[List[ImageAnnotationCreator]] = None
    #                                         # image_annotations: Optional[List[ImageAnnotation]] = None
    #                                     )
    #
    #                                     data_mapping[file_path] = output_file_path, [image_data_list], row.to_dict()
    #
    #     # Generate data summaries at the voyage and platform levels, and an iFDO at the deployment level
    #     summary_directories = set()
    #     ifdo_directories = set()
    #
    #     # Collect output directories
    #     for relative_dst, image_data_list, _ in data_mapping.values():
    #         if image_data_list:
    #             parts = relative_dst.parts
    #             if len(parts) > 0:
    #                 summary_directories.add(parts[0])
    #                 ifdo_directories.add(parts[0])
    #             if len(parts) > 1:
    #                 summary_directories.add(str(Path(parts[0]) / parts[1]))
    #                 ifdo_directories.add(str(Path(parts[0]) / parts[1]))
    #
    #     # Convert the set to a sorted list
    #     summary_directories = sorted(summary_directories)
    #     ifdo_directories = sorted(ifdo_directories)
    #
    #     # Subset the data_mapping to include only files in the summary directories
    #     for directory in summary_directories:
    #         subset_data_mapping = {
    #             src.as_posix(): image_data_list
    #             for src, (relative_dst, image_data_list, _) in data_mapping.items()
    #             if str(relative_dst).startswith(directory) and image_data_list
    #         }
    #
    #         # Create a dataset summary for each of these
    #         dataset_wrapper = DatasetWrapper(data_dir / directory, version=None, dry_run=True)
    #         dataset_wrapper.dry_run = False
    #         dataset_wrapper.generate_dataset_summary(subset_data_mapping, progress=False)
    #
    #         # Add the summary to the dataset mapping
    #         output_file_path = dataset_wrapper.summary_path.relative_to(data_dir)
    #         data_mapping[dataset_wrapper.summary_path] = output_file_path, None, None
    #
    #     # Subset the data_mapping to include only files in the ifdo directories
    #     for directory in ifdo_directories:
    #         subset_data_mapping = {
    #             relative_dst.relative_to(directory).as_posix(): image_data_list
    #             for src, (relative_dst, image_data_list, _) in data_mapping.items()
    #             if str(relative_dst).startswith(directory) and image_data_list
    #         }
    #
    #         # Create a iFDO for each of these
    #         dataset_wrapper = DatasetWrapper(data_dir / directory, version=None, dry_run=True)
    #         dataset_wrapper.dry_run = False
    #         dataset_wrapper.generate_ifdo(directory, subset_data_mapping, progress=False)
    #
    #         # Add the iFDO to the dataset mapping
    #         output_file_path = dataset_wrapper.metadata_path.relative_to(data_dir)
    #         data_mapping[dataset_wrapper.metadata_path] = output_file_path, None, None
    #
    #     return data_mapping

    def _get_station_metadata(self, data_dir: Path, site_id: str) -> tuple[str, float, float, float]:
        """Extract station metadata from the CSV file."""
        month, year = re.search(r"\d+([A-Za-z]+)(\d+)", data_dir.parent.name).groups()
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

    def _create_image_data(self, row: dict, image_datetime: str, latitude: float, longitude: float,
                           depth: float) -> ImageData:
        """Create an ImageData object for a given row."""
        image_creators = [
            ImagePI(name="Joanna Strzelecki", orcid="0000-0003-1138-2932"),
            ImagePI(name="Chris Jackett", orcid="0000-0003-1132-1558"),
        ]
        image_pi = ImagePI(name="Joanna Strzelecki", orcid="0000-0003-1138-2932")

        return ImageData(
            image_datetime=image_datetime,
            image_latitude=latitude,
            image_longitude=longitude,
            image_altitude=depth,
            image_coordinate_reference_system="EPSG:4326",
            image_platform=self.config.get("platform_id"),
            image_uuid=str(uuid.UUID(row["UUID"])),
            image_pi=image_pi,
            image_creators=image_creators,
            image_license="CC BY-NC 4.0",
            image_copyright="CSIRO",
            image_acquisition=ImageAcquisition.PHOTO,
            image_quality=ImageQuality.PRODUCT,
            image_deployment=ImageDeployment.SURVEY,
            image_navigation=ImageNavigation.SATELLITE,
            image_illumination=ImageIllumination.ARTIFICIAL_LIGHT,
            image_pixel_mag=ImagePixelMagnitude.UM,
            image_marine_zone=ImageMarineZone.WATER_COLUMN,
            image_spectral_resolution=ImageSpectralResolution.RGB,
            image_fauna_attraction=ImageFaunaAttraction.NONE,
            image_datetime_format="%Y-%m-%d %H:%M:%S±HHMM",
            image_target_environment="Pelagic",
            image_item_identification_scheme="<platform_id>_<site_id>_<replicate_id>_<objective_magnification>_<field_of_view>_<datetimestamp>_<capture_id>.<ext>",
            image_curation_protocol=f"Processed with Marimba {__version__}",
        )

    def _generate_summaries(
            self,
            data_mapping: dict[Path, tuple[Path, list[ImageData] | None, dict[str, Any] | None]],
            data_dir: Path,
    ) -> None:
        """Generate dataset summaries and iFDOs."""
        summary_directories = set()
        ifdo_directories = set()

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
        summary_directories = sorted(summary_directories)
        ifdo_directories = sorted(ifdo_directories)

        # Generate summaries
        for directory in summary_directories:
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
        for directory in ifdo_directories:
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
        rep_mapping = {}

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
                    row,
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
            **kwargs: dict,  # noqa: ARG002
    ) -> dict[Path, tuple[Path, list[ImageData] | None, dict[str, Any] | None]]:
        """Package the data directory into a standardized format."""
        data_mapping = {}

        # Get station metadata
        image_datetime, latitude, longitude, depth = self._get_station_metadata(data_dir, config.get("site_id"))

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
