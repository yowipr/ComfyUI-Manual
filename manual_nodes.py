import json
import torch
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
from io import BytesIO
import base64

from server import PromptServer, BinaryEventTypes



BASE_DIR = Path.cwd()
MANUAL_NODES_DIR = BASE_DIR.joinpath("ComfyUI", "custom_nodes", "Manual")


class AnyType(str):
  """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""

  def __ne__(self, __value: object) -> bool:
    return False


any = AnyType("*")

###############################################################################################################
class Layer:

    @classmethod
    def INPUT_TYPES(s):
        return{
            "required": {
                "layer": (any, {"default": None }),
            }
        }
    

    RETURN_TYPES = (any, "MASK")
    RETURN_NAMES = ("Layer", "Alpha")

    FUNCTION = "OnResult"

    CATEGORY = "Manual"

    def OnResult(self, layer):
        print(f"---------------------------------loading Layer {layer['type']}---------------------------------")
        #print("Type of layer:", type(layer))
        #print("Content of layer:", layer)
        layerValue = layer['value']


        if layer['type'] == "IMAGE":
            return self.Layer_IMAGE(layerValue)
        

        elif layer['type'] == "STRING":
            result = layerValue
            return (result,)
        
        elif layer['type'] == "LAYER":
            try:
                # Intentar deserializar la cadena JSON en un diccionario de Python
                layer_dict = json.loads(layerValue)
                img_data = layer_dict['ImageData']
                return self.Layer_IMAGE(img_data)
            except json.JSONDecodeError:
                print("Error decodificando JSON")
                return None
            except KeyError:
                print("La clave 'ImageData' no se encuentra en el JSON")
                return None


        else:
            return None
    

    def Layer_IMAGE(self, value):
        imgdata = base64.b64decode(value)
        result = Image.open(BytesIO(imgdata))

        if "A" in result.getbands():
            mask = np.array(result.getchannel("A")).astype(np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

        result = result.convert("RGB")
        result = np.array(result).astype(np.float32) / 255.0
        result = torch.from_numpy(result)[None,]
        return (result, mask)

    

###############################################################################################################

class Output:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                {
                    "result": (any, {}),
                }
             }
    

    RETURN_TYPES = ()
    FUNCTION = "OnResult"
    OUTPUT_NODE = True
    CATEGORY = "Manual"

    def OnResult(self, result):
        results = []
        for tensor in result:
            array = 255.0 * tensor.cpu().numpy()
            image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))

            server = PromptServer.instance
            server.send_sync(
                BinaryEventTypes.UNENCODED_PREVIEW_IMAGE,
                ["PNG", image, None],
                server.client_id,
            )
            
            results.append(
                {"source": "websocket", "content-type": "image/png", "type": "output"}
            )

        return {"ui": {"images": results}}




###############################################################################################################
class RenderArea:

    @classmethod
    def INPUT_TYPES(s):
        return{
            "required": {
                "layer": (any, {"default": None }),
            },
            "optional": {
                "source_layer": (any, {}),
                "dest_layer": (any, {}),
            }

        }
    

    RETURN_TYPES = (any,)
    RETURN_NAMES = ("Layer",)

    FUNCTION = "OnResult"

    CATEGORY = "Manual"

    def OnResult(self, layer, source_layer, dest_layer):
        print(f"---------------------------------loading RenderArea {layer['type']}---------------------------------")
        #print("Type of layer:", type(layer))
        #print("Content of layer:", layer)
        layerValue = layer['value']


        if layer['type'] == "IMAGE":
            return self.Layer_IMAGE(layerValue)
        

        elif layer['type'] == "STRING":
            result = layerValue
            return (result,)
        else:
            return None
    

    def Layer_IMAGE(self, value):
        imgdata = base64.b64decode(value)
        result = Image.open(BytesIO(imgdata))

        if "A" in result.getbands():
            mask = np.array(result.getchannel("A")).astype(np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

        result = result.convert("RGB")
        result = np.array(result).astype(np.float32) / 255.0
        result = torch.from_numpy(result)[None,]
        return (result, mask)

    