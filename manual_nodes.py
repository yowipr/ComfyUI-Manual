import torch
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
from io import BytesIO
import base64

from server import PromptServer, BinaryEventTypes

BASE_DIR = Path.cwd()
MANUAL_NODES_DIR = BASE_DIR.joinpath("ComfyUI", "custom_nodes", "Manual")

###############################################################################################################
class Layer:

    @classmethod
    def INPUT_TYPES(s):
        return{
            "required": {
                "image": ("STRING", {
                    "default": "null"
                }),
                
            }
        }
    


    RETURN_TYPES = ("IMAGE", "MASK")

    FUNCTION = "load_image"

    CATEGORY = "Manual"

    def load_image(self, image):
        imgdata = base64.b64decode(image)
        img = Image.open(BytesIO(imgdata))

        if "A" in img.getbands():
            mask = np.array(img.getchannel("A")).astype(np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

        img = img.convert("RGB")
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img)[None,]

        return (img, mask)

###############################################################################################################

class Output:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                {
                    "images": ("IMAGE",)
                }
             }
    

    RETURN_TYPES = ()
    FUNCTION = "send_images"
    OUTPUT_NODE = True
    CATEGORY = "Manual"

    def send_images(self, images):
        results = []
        for tensor in images:
            array = 255.0 * tensor.cpu().numpy()
            image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))

            server = PromptServer.instance
            server.send_sync(
                BinaryEventTypes.UNENCODED_PREVIEW_IMAGE,
                ["PNG", image, None],
                server.client_id,
            )
            results.append(
                # Could put some kind of ID here, but for now just match them by index
                {"source": "websocket", "content-type": "image/png", "type": "output"}
            )

        return {"ui": {"images": results}}


###############################################################################################################

