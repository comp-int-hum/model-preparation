from pathlib import Path
import sys
import re
import os
import os.path
import json
import argparse
import tempfile
import subprocess
import shutil
import shlex
from datetime import datetime
from glob import glob
import logging
from importlib import import_module
import importlib.util
import tokenize
import torch
import transformers
from ts.torch_handler.base_handler import BaseHandler
from ts.context import Context
from PIL import Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def transformer_prep(args, path):
    config = transformers.AutoConfig.from_pretrained(
        args.parameters_url
    )
    model = getattr(transformers, config.architectures[0]).from_pretrained(
        args.parameters_url,
        config=config
    )
    try:
        processor = getattr(transformers, "{}Processor".format(args.model_class_name)).from_pretrained(
            args.parameters_url
        )
    except:
        processor = None
    try:
        feature_extractor = transformers.AutoFeatureExtractor.from_pretrained(
            args.parameters_url
        )
    except:
        feature_extractor = None
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.parameters_url
        )
    except:
        tokenizer = None
    model.save_pretrained(path)
    if processor:
        processor.save_pretrained(path)
    if tokenizer:
        tokenizer.save_pretrained(path)        
    if feature_extractor:
        feature_extractor.save_pretrained(path)        
    return [f for f in glob(os.path.join(path, "*")) if not os.path.basename(f) == "pytorch_model.bin"]


def custom_prep(args):
    pass


if __name__== "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name", dest="model_name", required=True)
    parser.add_argument("--model_version", dest="model_version", default=1.0, required=True)
    parser.add_argument("--handler", dest="handler", default="object_detector", required=True)
    parser.add_argument("--dry_run", dest="dry_run", default=False, action="store_true")
    parser.add_argument("--requirements", dest="requirements", default=[], nargs="*")
    parser.add_argument("--input_data", dest="input_data")
    parser.add_argument("--output")

    subs = parser.add_subparsers(
        help="Alternative ways of specifying models"
    )
    
    transformer_parser = subs.add_parser(
        "transformers",
        help=""
    )
    transformer_parser.add_argument(
        "--model_class_name",
        dest="model_class_name",
        required=True
    )
    transformer_parser.add_argument(
        "--parameters_url",
        dest="parameters_url",
        required=True
    )
    #transformer_parser.add_argument(
    #    "--processor-class-name",
    #    dest="processor_class_name",
    #    required=True
    #)
    transformer_parser.set_defaults(
        prep_func=transformer_prep
    )

    custom_parser = subs.add_parser(
        "custom",
        help=""
    )
    custom_parser.add_argument(
        "--serialized-model",
        dest="serialized_model",
        required=True
    )
    transformer_parser.set_defaults(
        prep_func=transformer_prep #custom_prep
    )
    
    args = parser.parse_args()

    
    path = tempfile.mkdtemp()
    try:
        logging.info("Save model and other needed files under '%s'", path)
        fnames = args.prep_func(args, path)
        if args.requirements:
            with open(os.path.join(path, "requirements.txt"), "wt") as ofd:
                ofd.write("\n".join(args.requirements))
        os.mkdir(os.path.join(path, "MAR-INF"))
        with open(os.path.join(path, "MAR-INF/MANIFEST.json"), "wt") as ofd:
            ofd.write(
                json.dumps(
                    {
                        "createdOn": datetime.now().isoformat(),
                        "runtime": "python",
                        "model": {
                            "modelName": args.model_name,
                            "serializedFile": "pytorch_model.bin",
                            "handler": os.path.basename(args.handler),
                            "modelVersion": args.model_version
                        },
                        "archiverVersion": "0.6.0"                        
                    }
                )
            )
                              
        if args.dry_run:
            logger.info("Dry run of model")
            module_name = os.path.splitext(os.path.basename(args.handler))[0]
            spec = importlib.util.spec_from_file_location(
                module_name,
                args.handler
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            handler_classes = []
            for name in dir(module):
                item = getattr(module, name)
                try:
                    if issubclass(item, BaseHandler) and item.__module__ == module_name:
                        handler_classes.append(item)
                except:
                    logger.debug("Skipping '%s' as it's either not a handler or externally defined")
            handler_class = handler_classes[0]
            ctx = Context(
                model_name=args.model_name,
                model_dir=path,
                manifest=os.path.join(path, "MAR-INF/MANIFEST.json"),
                batch_size=1,
                gpu=False,
                mms_version=1.0
            )
            handler = handler_class()    
            handler.initialize(ctx)
            ctx.get_request_header = lambda x, y : False
            if args.input_data:
                with open(args.input_data, "rb") as ifd:
                    data = ifd.read()
                logger.info("Dry run response: %s", handler.handle([{"data" : data}], ctx))
        else:
            cmd = [
                "torch-model-archiver",
                "--model-name", args.model_name,
                "--version", args.model_version,
                "--serialized-file", "{}/pytorch_model.bin".format(path),
                "--handler", args.handler,
                "--extra-files", ",".join(fnames),
                "--export-path", path,
            ]
            if args.requirements:
                cmd += ["--requirements-file", os.path.join(path, "requirements.txt")]
            logging.info("invoking '%s'", shlex.join(cmd))
            pid = subprocess.Popen(cmd)
            pid.communicate()
            shutil.move("{}/{}.mar".format(path, args.model_name), args.output)

    except Exception as e:
        raise e
    finally:
        shutil.rmtree(path)

    
    sys.exit()

    logging.basicConfig(level=logging.INFO)
    logging.info("Transformers version %s", transformers.__version__)

    
    transformers.set_seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    logging.info("Download model and tokenizer '%s'", args.original_model_name)
    config = transformers.AutoConfig.from_pretrained(args.original_model_name, num_labels=args.num_labels, torchscript=(args.save_mode=="torchscript"))
    model = transformers.AutoModelForCausalLM.from_pretrained(args.original_model_name, config=config)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.original_model_name, do_lower_case=args.do_lower_case)
        
    path = tempfile.mkdtemp()
    try:
        logging.info("Save model and tokenizer model in '%s'", path)
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
        cmd = [
            "torch-model-archiver",
            "--model-name", args.model_name,
            "--version", args.model_version,
            "--serialized-file", "{}/pytorch_model.bin".format(path),
            "--handler", args.handler,
            "--extra-files", "{0}/config.json,{0}/special_tokens_map.json,{0}/tokenizer_config.json,{0}/tokenizer.json".format(path),
            "--export-path", path,
        ]
        logging.info("invoking '%s'", shlex.join(cmd))
        pid = subprocess.Popen(cmd)
        pid.communicate()
        shutil.move("{}/{}.mar".format(path, args.model_name), args.output)
    except Exception as e:
        raise e
    finally:
        shutil.rmtree(path)
