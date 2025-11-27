import os
import torch
import argparse

from models.lion_t5 import LIONT5InstructAdapter
from evaluation.img_cap_dataset import ImgCapDataset
from common.registry import registry
from omegaconf import OmegaConf
from pycocoevalcap.cider.cider import Cider
from tqdm import tqdm


def build_model(cfg):
    model_cfg = cfg.model
    model_cls = registry.get_model_class("lion_t5")
    return model_cls.from_config(model_cfg)


def build_datasets():
    coco_root = "images/coco"
    coco_img_path = os.path.join(coco_root, "images/val2014")
    coco_ann = "annotations/captions_val2014.json"
    coco_ann_path = os.path.join(coco_root, coco_ann)
    coco_dset = ImgCapDataset("coco", coco_img_path, coco_ann_path)

    textcaps_root = "images/textcaps"
    textcaps_img_path = os.path.join(textcaps_root, "images/train_images")
    textcaps_ann = "images/TextCaps_0.1_val.json"
    textcaps_ann_path = os.path.join(textcaps_root, textcaps_ann)
    textcaps_dset = ImgCapDataset("textcaps", textcaps_img_path, textcaps_ann_path)

    return coco_dset, textcaps_dset


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cfg-path",
        default="configs/eval.yaml",
        required=True,
        help="Path to evaluation config YAML",
    )
    return ap.parse_args()


def output_image_captions(model, dset: ImgCapDataset):
    question = "Please describe the image using a single short sentence."
    gts = {}
    res = {}

    for sample in tqdm(dset):
        img_id, img, ram_img, caption = sample
        gts[img_id] = [caption]
        hyp_caption = model.generate(
            {
                "image": img.unsqueeze(0),
                "ram_image": ram_img.unsqueeze(0),
                "question": [question],
                # "tags_for_dynamic_prompt": t1,
                "category": "image_level",
            }
        )
        res[img_id] = hyp_caption

    return gts, res


def main():
    cider = Cider()
    coco_dset, textcaps_dset = build_datasets()

    args = parse_args()
    cfg = OmegaConf.load(args.cfg_path)

    model = build_model(cfg)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    coco_gts, coco_res = output_image_captions(model, coco_dset)
    textcaps_gts, textcaps_res = output_image_captions(model, textcaps_dset)

    coco_cider = cider.compute_score(coco_gts, coco_res)
    textcaps_cider = cider.compute_score(textcaps_gts, textcaps_res)
    print(coco_cider[0])
    print(textcaps_cider[0])


if __name__ == "__main__":
    main()
