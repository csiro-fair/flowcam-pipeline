import os
import re
from datetime import datetime
from pathlib import Path
from shutil import copy2
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import piexif
import typer
from PIL import Image
from PIL.ExifTags import TAGS
from ifdo.models import ImageData
from marimba.core.pipeline import BasePipeline
from marimba.core.utils.rich import error_panel
from marimba.lib import image


class FlowCamPipeline(BasePipeline):
    """
    FlowCam pipeline
    """

    @staticmethod
    def get_pipeline_config_schema() -> dict:
        return {
        }

    @staticmethod
    def get_collection_config_schema() -> dict:
        return {
            "batch_id": "1a",
        }

    @staticmethod
    def natural_sort_key(s):
        """
        Extracts integers from a string and returns a tuple that can be used for sorting.
        """
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

    def find_sub_images(self, img_path: Path, output_dir: Path, start_i: int):
        # Load image and convert to grayscale
        img = cv2.imread(str(img_path))  # Convert Path object to string for cv2.imread
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Threshold image (black and white only)
        _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        # Find contours in the binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours by their position (top to bottom, left to right)
        contours = sorted(contours, key=lambda ctr: (cv2.boundingRect(ctr)[1], cv2.boundingRect(ctr)[0]))

        # Get the base filename without the extension
        base_filename = img_path.stem.replace(" ", "_").lower()

        i = start_i
        for contour in contours:
            # Get the rectangle that contains the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Ignore small regions
            if w * h > 100:

                # Extract sub-image
                sub_img = img[y:y + h, x:x + w]

                # Check if the standard deviation of pixel intensities is below a certain threshold
                if np.std(sub_img) < 80:
                    # Save the sub-image
                    output_path = output_dir / f"{base_filename}_{str(i).zfill(4)}.png"
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

            output_dir = source_path.parent / (source_path.name.replace("source", "work"))

            if not output_dir.exists():
                output_dir.mkdir(parents=True)

            sorted_file_names = sorted(source_path.iterdir(), key=lambda p: self.natural_sort_key(p.name))

            i = 1  # Initialize i outside the loop
            for file_path in sorted_file_names:
                if file_path.suffix.lower() == '.png':
                    self.logger.info(f"Extracting vignettes from {file_path.name}")
                    i = self.find_sub_images(file_path, output_dir, i)
                if file_path.suffix.lower() == '.csv':
                    if not self.dry_run:
                        copy2(file_path, data_dir)
                    self.logger.debug(f"Copied {file_path.resolve().absolute()} -> {data_dir}")

    def _process(self, data_dir: Path, config: Dict[str, Any], **kwargs: dict):

        # # Use a glob pattern to find all CSV files
        # csv_files = list(source_path.glob('*.csv'))
        #
        # # Check if there are any CSV files and get the first one
        # if csv_files:
        #     data_csv = csv_files[0]
        #     self.logger.info(f"Found data CSV file: {data_csv}")
        # else:
        #     error_message = f"No CSV files found in the {str(source_path)} directory."
        #     self.logger.error(error_message)
        #     print(error_panel(error_message))
        #     raise typer.Exit()

    def _compose(
        self, data_dirs: List[Path], configs: List[Dict[str, Any]], **kwargs: dict
    ) -> Dict[Path, Tuple[Path, List[ImageData]]]:
        data_mapping = {}
        # for data_dir, config in zip(data_dirs, configs):
        #     file_paths = []
        #     file_paths.extend(data_dir.glob("**/*"))
        #     base_output_path = Path(config.get("deployment_id"))
        #
        #     sensor_data_df = pd.read_csv(next((data_dir / "data").glob("*.CSV")))
        #     sensor_data_df["FinalTime"] = pd.to_datetime(
        #         sensor_data_df["FinalTime"], format="%Y-%m-%d %H:%M:%S.%f"
        #     ).dt.floor("S")
        #
        #     for file_path in file_paths:
        #         output_file_path = base_output_path / file_path.relative_to(data_dir)
        #
        #         if (
        #             file_path.is_file()
        #             and file_path.suffix.lower() in [".jpg"]
        #             and "_THUMB" not in file_path.name
        #             and "overview" not in file_path.name
        #         ):
        #             iso_timestamp = file_path.name.split("_")[5]
        #             target_datetime = pd.to_datetime(
        #                 iso_timestamp, format="%Y%m%dT%H%M%SZ"
        #             )
        #             matching_rows = sensor_data_df[
        #                 sensor_data_df["FinalTime"] == target_datetime
        #             ]
        #
        #             if not matching_rows.empty:
        #                 # in iFDO, the image data list for an image is a list containing single ImageData
        #                 image_data_list = [
        #                     ImageData(
        #                         image_datetime=datetime.strptime(
        #                             iso_timestamp, "%Y%m%dT%H%M%SZ"
        #                         ),
        #                         image_latitude=matching_rows["UsblLatitude"].values[0],
        #                         image_longitude=float(
        #                             matching_rows["UsblLongitude"].values[0]
        #                         ),
        #                         image_depth=float(matching_rows["Altitude"].values[0]),
        #                         image_altitude=float(
        #                             matching_rows["Altitude"].values[0]
        #                         ),
        #                         image_event=str(matching_rows["Operation"].values[0]),
        #                         image_platform=self.config.get("platform_id"),
        #                         image_sensor=str(matching_rows["Camera"].values[0]),
        #                         image_camera_pitch_degrees=float(
        #                             matching_rows["Pitch"].values[0]
        #                         ),
        #                         image_camera_roll_degrees=float(
        #                             matching_rows["Roll"].values[0]
        #                         ),
        #                         image_uuid=str(uuid4()),
        #                         # image_pi=self.config.get("voyage_pi"),
        #                         image_creators=[],
        #                         image_license="MIT",
        #                         image_copyright="",
        #                         image_abstract=self.config.get("abstract"),
        #                     )
        #                 ]
        #
        #                 data_mapping[file_path] = output_file_path, image_data_list
        #
        #         elif file_path.is_file():
        #             data_mapping[file_path] = output_file_path, None

        return data_mapping
