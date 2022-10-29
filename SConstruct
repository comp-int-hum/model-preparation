import os
import os.path
import logging
import random
import subprocess
import shlex
import gzip
import re
import functools
import time
import imp
import sys
import json

# workaround needed to fix bug with SCons and the pickle module
del sys.modules['pickle']
sys.modules['pickle'] = imp.load_module('pickle', *imp.find_module('pickle'))
import pickle

# actual variable and environment objects
vars = Variables("custom.py")

vars.AddVariables(
    (
        "OUTPUT_WIDTH",
        "",
        1000
    ),
    (
        "MODELS",
        "",
        [
            {
                "MODEL_NAME" : "trocr_printed",
                "MODEL_VERSION" : 1.0,
                "HANDLER" : "scripts/ocr_handler.py",
                "STYLE" : "transformers",
                "MODEL_CLASS_NAME" : "TrOCR",
                "PARAMETERS_URL" : "microsoft/trocr-base-printed"
            },
            {
                "MODEL_NAME" : "trocr_handwritten",
                "MODEL_VERSION" : 1.0,
                "HANDLER" : "scripts/ocr_handler.py",
                "STYLE" : "transformers",
                "MODEL_CLASS_NAME" : "TrOCR",
                "PARAMETERS_URL" : "microsoft/trocr-base-handwritten"
            },
            {
                "MODEL_NAME" : "resnet50",
                "MODEL_VERSION" : 1.0,
                "HANDLER" : "scripts/object_detection_handler.py",
                "STYLE" : "transformers",
                "MODEL_CLASS_NAME" : "DetrForObjectDetection",
                "PARAMETERS_URL" : "facebook/detr-resnet-50",
            },
            {
                "MODEL_NAME" : "opt125m",
                "MODEL_VERSION" : 1.0,
                "HANDLER" : "scripts/text_generation_handler.py",
                "STYLE" : "transformers",
                "MODEL_CLASS_NAME" : "OPTForCausalLM",
                "PARAMETERS_URL" : "facebook/opt-125m"
            },
            {
                "MODEL_NAME" : "bloom560m",
                "MODEL_VERSION" : 1.0,
                "HANDLER" : "scripts/text_generation_handler.py",
                "STYLE" : "transformers",
                "MODEL_CLASS_NAME" : "BloomForCausalLM",
                "PARAMETERS_URL" : "bigscience/bloom-560m"
            }
        ]
    ),
)

env = Environment(
    variables=vars,
    ENV=os.environ,
    TARFLAGS="-c -z",
    TARSUFFIX=".tgz",
    tools=["default"],
    BUILDERS={
        "PackageModel" : Builder(
            action="python scripts/package_model.py --model_name ${MODEL_NAME} --model_version ${MODEL_VERSION} --handler ${HANDLER} --requirements ${REQUIREMENTS} --output ${TARGETS[0]} ${STYLE} --model_class_name ${MODEL_CLASS_NAME} --parameters_url ${PARAMETERS_URL}"
        )
    }
)

# function for width-aware printing of commands
def print_cmd_line(s, target, source, env):
    if len(s) > int(env["OUTPUT_WIDTH"]):
        print(s[:int(float(env["OUTPUT_WIDTH"]) / 2) - 2] + "..." + s[-int(float(env["OUTPUT_WIDTH"]) / 2) + 1:])
    else:
        print(s)

# and the command-printing function
env['PRINT_CMD_LINE_FUNC'] = print_cmd_line

# and how we decide if a dependency is out of date
env.Decider("timestamp-newer")

#image_classifier
#object_detector
#text_classifier
#image_segmenter

for model in env["MODELS"]:
    pkg = env.PackageModel(
        os.path.join("work/{}.mar".format(model["MODEL_NAME"])),
        [],
        **model
    )
    if os.path.exists(model["HANDLER"]):
        env.Depends(pkg, [model["HANDLER"], "scripts/package_model.py"])
