import os
import json

from torch.utils.data import Dataset
from PIL import Image
from ram.transform import get_transform
from preprocessors.lion_preprocessors import ImageEvalProcessor


class ImgCapDataset(Dataset):
    def __init__(self, dset_name: str, img_path: str, ann_path: str):
        super().__init__()
        self.img_path = img_path
        self.ann_path = ann_path

        if dset_name == "coco":
            self._load_coco_ann_file()
        elif dset_name == "textcaps":
            self._load_textcaps_ann_file()
        else:
            raise KeyError("Dataset name error")

        self.processor = ImageEvalProcessor()
        self.ram_processor = get_transform()

    def _load_coco_ann_file(self):
        with open(self.ann_path, "r") as f:
            data = json.load(f)

            id_to_fname = {img["id"]: img["file_name"] for img in data["images"]}
            samples = []
            for ann in data["annotations"]:
                caption = ann["caption"]
                img_id = ann["image_id"]
                file_name = id_to_fname[img_id]
                img_path = os.path.join(self.img_path, file_name)
                # if not os.path.exists(img_path):
                #     raise RuntimeError(f"File not matching: {img_path}")
                samples.append((img_id, img_path, caption))
            self.samples = samples

    def _load_textcaps_ann_file(self):
        with open(self.ann_path, "r") as f:
            data = json.load(f)
            data = data["data"]

            samples = []
            for info in data:
                img_id = info["image_id"]
                img_path = info["image_path"]
                img_path = os.path.join(self.img_path, img_path)
                # if not os.path.exists(img_path):
                #     raise RuntimeError(f"File not matching: {img_path}")
                caption = info["caption_str"]
                samples.append((img_id, img_path, caption))
            self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, img_path, caption = self.samples[idx]
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img = self.processor(img)
            ram_img = self.ram_processor(img)
            sample = (img_id, img, ram_img, caption)

        return sample


if __name__ == "__main__":
    coco_root = "images/coco"
    coco_img_path = os.path.join(coco_root, "images/val2014")
    coco_ann = "annotations/captions_val2014.json"
    coco_ann_path = os.path.join(coco_root, coco_ann)

    textcaps_root = "images/textcaps"
    textcaps_img_path = os.path.join(textcaps_root, "images/train_images")
    textcaps_ann = "images/TextCaps_0.1_val.json"
    textcaps_ann_path = os.path.join(textcaps_root, textcaps_ann)

    # coco = ImgCapDataset("coco", coco_img_path, coco_ann_path)
    textcaps_dset = ImgCapDataset("textcaps", coco_img_path, coco_ann_path)
