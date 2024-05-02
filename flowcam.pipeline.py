import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from shutil import copy2
from typing import Any, Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import typer
from ifdo.models import (
    ImageData,
    ImagePI,
    ImageAcquisition,
    ImageQuality,
    ImageDeployment,
    ImageNavigation,
    ImageIllumination,
    ImagePixelMagnitude,
    ImageMarineZone,
    ImageSpectralResolution,
    ImageCaptureMode,
    ImageFaunaAttraction,
)

from marimba.core.pipeline import BasePipeline
from marimba.core.utils.rich import error_panel


class FlowCamPipeline(BasePipeline):
    """
    FlowCam pipeline
    """

    @staticmethod
    def get_pipeline_config_schema() -> dict:
        return {
            "project_pi": "Joanna Strzelecki",
            "data_collector": "Joanna Strzelecki",
            "platform_id": "FC8000",
        }

    @staticmethod
    def get_collection_config_schema() -> dict:
        return {
            "site_id": "CS6",
        }

    @staticmethod
    def natural_sort_key(s):
        """
        Extracts integers from a string and returns a tuple that can be used for sorting.
        """
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]

    def find_sub_images(self, img_path: Path, images_dir: Path, start_i: int):
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
            if w * h > 100:
                # Extract sub-image
                sub_img = img[y : y + h, x : x + w]

                # Check if the standard deviation of pixel intensities is below a certain threshold
                if np.std(sub_img) < 80:
                    # Save the sub-image
                    output_path = images_dir / f"{base_filename}_{str(i).zfill(6)}.jpg"
                    cv2.imwrite(str(output_path), sub_img)  # Convert Path object to string for cv2.imwrite
                    i += 1

        return i

    def _import(
        self,
        data_dir: Path,
        source_paths: List[Path],
        config: Dict[str, Any],
        **kwargs: dict,
    ):
        # This is really a process to extract the individual vignettes from the collage images
        self.logger.info(f"Importing data from {source_paths=} to {data_dir}")
        for source_path in source_paths:
            if not source_path.is_dir():
                continue

            match = re.search(r"rep(\d+)", str(source_path))
            if match:
                rep_id = int(match.group(1))  # Convert the matched group to integer
                self.logger.debug(f"Rep ID found in the source path is: {rep_id}")
            else:
                self.logger.warning("No Rep ID found in the source path.")
                continue

            rep_dir = data_dir / config.get("site_id") / str(rep_id).zfill(2)
            rep_dir.mkdir(exist_ok=True, parents=True)

            sorted_file_names = sorted(source_path.iterdir(), key=lambda p: self.natural_sort_key(p.name))

            i = 1  # Initialize i outside the loop
            for file_path in sorted_file_names:
                if file_path.suffix.lower() == ".png":
                    self.logger.info(f"Extracting vignettes from {file_path.name}")
                    images_dir = rep_dir / "images"
                    images_dir.mkdir(exist_ok=True, parents=True)
                    i = self.find_sub_images(file_path, images_dir, i)
                if file_path.suffix.lower() == ".tiff":
                    self.logger.info(f"Extracting vignettes from {file_path.name}")
                    cal_dir = rep_dir / "cal"
                    cal_dir.mkdir(exist_ok=True, parents=True)
                    copy2(file_path, cal_dir)
                if file_path.suffix.lower() in [".csv", ".pdf"]:
                    if not self.dry_run:
                        rep_data_dir = rep_dir / "data"
                        rep_data_dir.mkdir(exist_ok=True, parents=True)
                        copy2(file_path, rep_data_dir)
                    self.logger.debug(f"Copied {file_path.resolve().absolute()} -> {rep_dir}")

    def _process(self, data_dir: Path, config: Dict[str, Any], **kwargs: dict):
        for site_id in data_dir.iterdir():
            for rep_id in site_id.iterdir():
                rep_dir = data_dir / site_id.name / rep_id.name
                rep_data_dir = rep_dir / "data"
                image_dir = rep_dir / "images"

                try:
                    data_csv = rep_data_dir / f"{data_dir.parent.name}rep{int(rep_id.name)}.csv"
                    summary_csv = rep_data_dir / f"{data_dir.parent.name}rep{int(rep_id.name)}_summary.csv"
                    self.logger.info(f"Found data CSV file: {data_csv}")
                except Exception:
                    error_message = f"No CSV files found in the {str(rep_data_dir)} directory."
                    self.logger.error(error_message)
                    print(error_panel(error_message))
                    raise typer.Exit()

                def extract_magnification(file_path):
                    with open(file_path, "r") as file:
                        content = file.read()

                    # Using regular expression to find 'Magnification: value'
                    match = re.search(r"Magnification,\s*([\d.]+)", content)
                    if match:
                        return match.group(1)  # Return the value found after 'Magnification:'
                    else:
                        return "Magnification value not found"

                data_df = pd.read_csv(data_csv)
                data_df["Filename"] = ""
                objective_mag = extract_magnification(summary_csv)

                # Loop through each row in the DataFrame
                for index, row in data_df.iterrows():
                    capture_id = str(row["Capture ID"]).zfill(6)

                    image_filename = f"{row['Name']}_{capture_id}.jpg"
                    image_path = image_dir / image_filename

                    # TODO: Double check timezone
                    creation_date = datetime.strptime(row["Timestamp"], "%m-%d-%Y %H:%M:%S")
                    iso_timestamp = creation_date.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

                    output_filename = (
                        f'{self.config.get("platform_id")}_'
                        + f"{site_id.name}_"
                        + f"{rep_id.name}_"
                        + f"{objective_mag}X_"
                        + f"{iso_timestamp}_"
                        + f"{capture_id}"
                        + f".JPG"
                    )
                    output_path = image_dir / output_filename

                    # Store the new filename in the DataFrame
                    data_df.at[index, "Filename"] = Path(output_path).relative_to(data_dir)

                    if not self.dry_run and image_path.exists():
                        os.rename(image_path, output_path)

                # Save the modified DataFrame
                data_df.to_csv(rep_data_dir / f"{data_dir.parent.name}rep{int(rep_id.name)}.csv", index=False)

    def _compose(self, data_dirs: List[Path], configs: List[Dict[str, Any]], **kwargs: dict) -> Dict[Path, Tuple[Path, List[ImageData], Dict]]:
        data_mapping = {}

        for data_dir, config in zip(data_dirs, configs):
            for site_id in data_dir.iterdir():
                for rep_id in site_id.iterdir():
                    rep_dir = data_dir / site_id.name / rep_id.name
                    rep_data_dir = rep_dir / "data"

                    all_files = data_dir.glob("**/*")

                    # Filter out files that are in or under any 'images' directory
                    ancillary_files = [f for f in all_files if "images" not in f.parts]

                    # Add ancillary files to data mapping
                    for file_path in ancillary_files:
                        if file_path.is_file():
                            output_file_path = file_path.relative_to(data_dir)
                            data_mapping[file_path] = output_file_path, None, None

                    try:
                        data_csv = rep_data_dir / f"{data_dir.parent.name}rep{int(rep_id.name)}.csv"
                        self.logger.info(f"Found data CSV file: {data_csv}")
                    except Exception:
                        error_message = f"No CSV files found in the {str(rep_data_dir)} directory."
                        self.logger.error(error_message)
                        print(error_panel(error_message))
                        raise typer.Exit()

                    data_df = pd.read_csv(data_csv)

                    for index, row in data_df.iterrows():
                        file_path = data_dir / row["Filename"]
                        output_file_path = file_path.relative_to(data_dir)

                        print(file_path, file_path.is_file())

                        if file_path.is_file() and file_path.suffix.lower() in [".jpg"]:
                            image_creators = [
                                ImagePI(name="Joanna Strzelecki", orcid="0000-0000-0000-0000"),
                                ImagePI(name="Chris Jackett", orcid="0000-0003-1132-1558"),
                                ImagePI(name="CSIRO", orcid=""),
                            ]
                            image_pi = ImagePI(name="Joanna Strzelecki", orcid="0000-0000-0000-0000")

                            # in iFDO, the image data list for an image is a list containing single ImageData
                            image_data_list = [
                                ImageData(
                                    # iFDO core (required)
                                    image_datetime=datetime.strptime(row["Timestamp"], "%m-%d-%Y %H:%M:%S"),
                                    # image_latitude=float(row["latitude"]),
                                    # image_longitude=float(row["longitude"]),
                                    # image_altitude=None,
                                    # image_coordinate_reference_system="EPSG:4326",
                                    # image_coordinate_uncertainty_meters=None,
                                    # image_context=row["image_context"],
                                    # image_project=row["survey_id"],
                                    # image_event=f'{row["survey_id"]}_{row["deployment_number"]}',
                                    image_platform=self.config.get("platform_id"),
                                    # image_sensor=str(row["camera_name"]).strip(),
                                    image_uuid=str(uuid.UUID(row["UUID"])),
                                    # Note: Marimba automatically calculates and injects the SHA256 hash during packaging
                                    # image_hash_sha256=image_hash_sha256,
                                    image_pi=image_pi,
                                    image_creators=image_creators,
                                    image_license="CC BY-NC-ND 4.0",
                                    image_copyright="CSIRO",
                                    # image_abstract=row["abstract"],
                                    #
                                    # # iFDO capture (optional)
                                    image_acquisition=ImageAcquisition.PHOTO,
                                    image_quality=ImageQuality.PRODUCT,
                                    image_deployment=ImageDeployment.SURVEY,
                                    # image_navigation=ImageNavigation.RECONSTRUCTED,
                                    # TODO: Mention to Timm Schoening
                                    # TODO: Also ask about mapping to EXIF
                                    # image_scale_reference=ImageScaleReference.NONE,
                                    image_illumination=ImageIllumination.ARTIFICIAL_LIGHT,
                                    image_pixel_mag=ImagePixelMagnitude.MM,
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
                                    image_overlap_fraction=0,
                                    image_datetime_format="%m-%d-%Y %H:%M:%S",
                                    # image_camera_pose: Optional[CameraPose] = None
                                    # image_camera_housing_viewport=camera_housing_viewport,
                                    # image_flatport_parameters: Optional[FlatportParameters] = None
                                    # image_domeport_parameters: Optional[DomeportParameters] = None
                                    # image_camera_calibration_model: Optional[CameraCalibrationModel] = None
                                    # image_photometric_calibration: Optional[PhotometricCalibration] = None
                                    # image_objective: Optional[str] = None
                                    image_target_environment="Benthic habitat",
                                    # image_target_timescale: Optional[str] = None
                                    # image_spatial_constraints: Optional[str] = None
                                    # image_temporal_constraints: Optional[str] = None
                                    # image_time_synchronization: Optional[str] = None
                                    image_item_identification_scheme="<platform_id>_<site_id>_<replicate_id>_<objective_magnification>_<datetimestamp>_<capture_id>.<ext>",
                                    image_curation_protocol="Processed with Marimba v0.3",
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
                            ]

                            data_mapping[file_path] = output_file_path, image_data_list, row.to_dict()

        return data_mapping
