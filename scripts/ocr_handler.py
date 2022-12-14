from abc import ABC
import json
import base64
import logging
import os
import torch
import io
from PIL import Image
from torchvision import transforms
import transformers
from ts.torch_handler.base_handler import BaseHandler
from ts.context import Context
import requests


logger = logging.getLogger(__name__)


class CustomHandler(BaseHandler, ABC):
    initialized = False
    image_processing = transforms.Compose([transforms.ToTensor()])

    def initialize(self, ctx):
        self.manifest = ctx.manifest

        # if the manifest is a string pointing to an existing file, load
        # that file's contents as a JSON object and treat it as the manifest
        if isinstance(ctx.manifest, str) and os.path.exists(ctx.manifest):
            with open(ctx.manifest) as f:
                self.manifest = json.load(f)

        properties = ctx.system_properties

        # this (probably) defaults to something equivalent to "." (the current
        # directory) if no such property exists
        model_dir = properties.get("model_dir")

        # resolve the full path to the serialized model
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)

        # decide which device (cpu, or a gpu) the model will run on
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )

        # instantiate the model
        config = transformers.AutoConfig.from_pretrained(
            model_dir
        )
        self.model = getattr(transformers, config.architectures[0]).from_pretrained(            
            model_dir,
            config=config
        )
        self.processor = transformers.TrOCRProcessor.from_pretrained(
            model_dir
        )
        self.model.eval()
        self.initialized = True

    def handle(self, data, context):
        retval = []
        images = []
        for row in data:
            image = Image.open(io.BytesIO(row.get("data"))).convert("RGB")
            pixel_values = self.processor(image, return_tensors="pt").pixel_values 
            generated_ids = self.model.generate(pixel_values)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            retval.append(
                {
                    "bounding_box" : [0, 0, image.size[0], image.size[1]],
                    "text" : generated_text,
                    "probability" : 1.0
                }
            )
        return retval
