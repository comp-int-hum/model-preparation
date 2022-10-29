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
        )
        try:
            self.feature_extractor = transformers.AutoFeatureExtractor.from_pretrained(
                model_dir
            )
        except:
            self.feature_extractor = None
        try:
            self.processor = transformers.AutoProcessor.from_pretrained(
                model_dir
            )
        except:
            self.processor = None
        self.model.eval()
        self.initialized = True


    def handle(self, data, context):
        retval = []
        images = []
        for row in data:
            image = Image.open(io.BytesIO(row.get("data")))
            inputs = self.feature_extractor(image, return_tensors="pt")
            outputs = self.model(**inputs)
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.feature_extractor.post_process(outputs, target_sizes=target_sizes)[0]                
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                retval.append(
                    {
                        "score" : score.item(),
                        "label" : self.model.config.id2label[label.item()],
                        "bounding_box" : box
                    }
                )
        return [retval]
