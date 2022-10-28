from abc import ABC
import json
import logging
import os
import torch
import transformers
from ts.torch_handler.base_handler import BaseHandler
from ts.context import Context


logger = logging.getLogger(__name__)


class CustomHandler(BaseHandler, ABC):
    initialized = False
    #    def __init__(self):
    #        super(CustomHandler, self).__init__()
    #        self.initialized = False

    def initialize(self, ctx):
        """Load the model and any other preliminary steps needed.

        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model and its parameters.
        """
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
        self.model.eval()
        self.initialized = True


    def handle(self, data, context):
        """Entry point for default handler. It takes the data from the input request and returns
           the predicted outcome for the input.
        Args:
            data (list): The input data that needs to be made a prediction request on.
            context (Context): It is a JSON Object containing information pertaining to
                               the model artefacts parameters.
        Returns:
            list : Returns a list of dictionary with the predicted response.
        """
        return data
        # It can be used for pre or post processing if needed as additional request
        # information is available in context
        #start_time = time.time()

        self.context = context
        metrics = self.context.metrics

        is_profiler_enabled = os.environ.get("ENABLE_TORCH_PROFILER", None)
        if is_profiler_enabled:
            if PROFILER_AVAILABLE:
                output, _ = self._infer_with_profiler(data=data)
            else:
                raise RuntimeError(
                    "Profiler is enabled but current version of torch does not support."
                    "Install torch>=1.8.1 to use profiler."
                )
        else:
            if self._is_describe():
                output = [self.describe_handle()]
            else:
                data_preprocess = self.preprocess(data)

                if not self._is_explain():
                    output = self.inference(data_preprocess)
                    output = self.postprocess(output)
                else:
                    output = self.explain_handle(data_preprocess, data)

        #stop_time = time.time()
        #metrics.add_time(
        #    "HandlerTime", round((stop_time - start_time) * 1000, 2), None, "ms"
        #)
        return output
