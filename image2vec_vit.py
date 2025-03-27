import logging
import os
import threading

import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTModel

from download_vit import MODEL_NAME

hf_home = os.getenv("HF_HOME", None)
if hf_home:
    os.makedirs(hf_home, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


class Img2VecViT:
    def __init__(self, cuda_support, cuda_core):
        self.device = torch.device(cuda_core if cuda_support else "cpu")

        self.model = ViTModel.from_pretrained(
            MODEL_NAME,
            cache_dir=hf_home,
        )

        self.layer_output_size = self.model.config.hidden_size

        if self.layer_output_size != 768:
            raise ValueError(
                "Only ViT models with hidden size of 768 are supported at the moment"
            )

        self.model = self.model.to(self.device)  # type: ignore
        self.model.eval()

        self.processor = ViTImageProcessor.from_pretrained(
            MODEL_NAME,
            cache_dir=hf_home,
        )
        self.lock = threading.Lock()

    def _process_inputs(self, image: Image.Image):
        rgb_image = image.convert("RGB")
        """
        If one of the image dimensions is 1 or 3 it can confuse the `infer_channel_dimension_format` function
        in the `Transformers` library, so we set the input_data_format to 'channels_last' in that case.
        """
        input_data_format = (
            "channels_last"
            if rgb_image.width in (1, 3) or rgb_image.height in (1, 3)
            else None
        )
        try:
            inputs = self.processor(
                images=rgb_image,
                return_tensors="pt",
                input_data_format=input_data_format,
            )
        except ValueError:
            """
            The conversion of a PIL.Image.Image to a numpy array should yield the following shape:
            (width, height, channels) so long as we are calling .convert("RGB") on the image beforehand.
            Therefore, this except block should never trigger but I am leaving it here as a precaution for
            if we ever do change the image conversion method.
            """
            logger.error(
                "Unable to infer color channel format, defaulting to 'channels_first'"
            )
            inputs = self.processor(
                images=rgb_image,
                return_tensors="pt",
                input_data_format="channels_first",
            )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        return inputs

    def get_vec(self, image: Image.Image):
        inputs = self._process_inputs(image)

        with self.lock:
            with torch.no_grad():
                outputs = self.model(**inputs)
                features = outputs.last_hidden_state.mean(dim=1)

        return features.detach().cpu().numpy()[0]
