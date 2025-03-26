import base64
from io import BytesIO

from PIL import Image
import numpy as np
from pydantic import BaseModel

from image2vec_vit import Img2VecViT


class VectorImagePayload(BaseModel):
    id: str
    image: str


class ImageVectorizer:
    img2vec: Img2VecViT

    def __init__(self, cuda_support, cuda_core):
        self.img2vec = Img2VecViT(cuda_support, cuda_core)

    async def vectorize(self, image_base64: str) -> np.ndarray:
        try:
            image = self.base64_to_pillow(image_base64)
            if image is None:
                raise ValueError("Invalid image data")
            return self.img2vec.get_vec(image)
        except (RuntimeError, TypeError, NameError, Exception) as e:
            print("vectorize error:", e)
            raise e
        finally:
            del image

    def base64_to_pillow(self, image_base64: str):
        try:
            file_content = base64.b64decode(image_base64)
            return Image.open(BytesIO(file_content))
        except Exception as e:
            print(str(e))
            return None
