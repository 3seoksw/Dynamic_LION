import os
import torch
import argparse

from models.lion_t5 import LIONT5InstructAdapter
from evaluation.vqa_dataset import VQADataset
from common.registry import registry
from omegaconf import OmegaConf
from pycocoevalcap.cider.cider import Cider
from tqdm import tqdm


def build_model(cfg):
    model_cfg = cfg.model
    model_cls = registry.get_model_class("lion_t5")
    return model_cls.from_config(model_cfg)


def build_datasets():
    aokvqa_root = "images/aokvqa"
    aokvqa_img_path = os.path.join(aokvqa_root, "coco/test2017")
    aokvqa_q_path = os.path.join(aokvqa_root, "images/aokvqa/aokvqa_v1p0_test.json")
    aokvqa_dset = VQADataset("okvqa", aokvqa_img_path, aokvqa_q_path, "")

    okvqa_root = "images/okvqa"
    okvqa_img_path = os.path.join(okvqa_root, "images/val2014")
    okvqa_q_path = os.path.join(
        okvqa_root, "images/OpenEnded_mscoco_val2014_questions.json"
    )
    okvqa_ann_path = os.path.join(okvqa_root, "images/mscoco_val2014_annotations.json")
    okvqa_dset = VQADataset("okvqa", okvqa_img_path, okvqa_q_path, okvqa_ann_path)

    return okvqa_dset, aokvqa_dset


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cfg-path",
        default="configs/eval.yaml",
        required=True,
        help="Path to evaluation config YAML",
    )
    return ap.parse_args()


def output_answer(model, dset: VQADataset):
    total = 0
    correct = 0
    for sample in tqdm(dset):
        total += 1
        q_data, img, ram_img = sample
        question = q_data["question"]
        answers = q_data["top-1"]
        pred = model.generate(
            {
                "image": img.unsqueeze(0),
                "ram_img": ram_img.unsqueeze(0),
                "question": [question],
                # "tags_for_dynamic_prompt": t1,
                "category": "image_level",
            }
        )
        if pred in answers:
            correct += 1

    return correct / total


def main():
    okvqa_dset, aokvqa_dset = build_datasets()

    args = parse_args()
    cfg = OmegaConf.load(args.cfg_path)

    model = build_model(cfg)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    okvqa_acc = output_answer(model, okvqa_dset)
    aokvqa_acc = output_answer(model, aokvqa_dset)

    print(okvqa_acc, aokvqa_acc)
