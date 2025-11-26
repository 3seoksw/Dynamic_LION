import os
import json

from collections import Counter
from torch.utils.data import Dataset
from PIL import Image

from ram.transform import get_transform
from preprocessors.lion_preprocessors import ImageEvalProcessor


class VQADataset(Dataset):
    def __init__(
        self,
        dset_name: str,
        img_path: str,
        question_path: str,
        ann_path: str,
        max_len: int = 1000,
    ):
        super().__init__()
        self.img_path = img_path
        self.question_path = question_path
        self.ann_path = ann_path
        self.max_len = max_len

        self.samples = []
        if dset_name == "okvqa":
            self._load_OKVQA_dset()
        elif dset_name == "aokvqa":
            self._load_AOKVQA_dset()
        else:
            raise KeyError("Dataset name error")

        self.processor = ImageEvalProcessor()
        self.ram_processor = get_transform()

    def _load_AOKVQA_dset(self):
        with open(self.question_path, "r") as f:
            aokvqa = json.load(f)
            for data in aokvqa:
                q_id = data["question_id"]
                question = data["question"]
                answers = data["choices"]
                img_id = data["image_id"]
                img_filename = f"{img_id:012d}.jpg"
                img_path = os.path.join(self.img_path, img_filename)

                sample = {
                    "question_id": q_id,
                    "question": question,
                    "image_id": img_id,
                    "image_path": img_path,
                    "answers": answers,
                    "top-1": answers,
                }
                self.samples.append(sample)

    def _load_OKVQA_dset(self):
        q_data = {}
        with open(self.question_path, "r") as f:
            data = json.load(f)
            for q in data["questions"]:
                q_id = q["question_id"]
                img_id = q["image_id"]
                q_str = q["question"]
                q_data[q_id] = {"question": q_str, "image_id": img_id}

        with open(self.ann_path, "r") as f:
            data = json.load(f)
            for ann in data["annotations"]:
                q_id = ann["question_id"]
                if q_id not in q_data:
                    continue
                answers = [a["answer"] for a in ann["answers"]]
                counts = Counter(answers)
                max_count = max(counts.values())
                top_answers = [ans for ans, c in counts.items() if c == max_count]

                img_id = q_data[q_id]["image_id"]
                img_filename = f"COCO_val2014_{img_id:012d}.jpg"
                img_path = os.path.join(self.img_path, img_filename)

                sample = {
                    "question_id": q_id,
                    "question": q_data[q_id]["question"],
                    "image_id": img_id,
                    "image_path": img_path,
                    "answers": answers,
                    "top-1": top_answers,
                }
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        with Image.open(sample["image_path"]) as img:
            img = img.convert("RGB")
            proc_img = self.processor(img)
            ram_img = self.ram_processor(img)
        return sample, proc_img, ram_img


if __name__ == "__main__":
    okvqa = VQADataset(
        dset_name="aokvqa",
        img_path="images/aokvqa/coco/test2017",
        question_path="images/aokvqa/aokvqa_v1p0_test.json",
        ann_path="",
    )
    okvqa = VQADataset(
        "okvqa",
        "images/okvqa/images/val2014/",
        "images/okvqa/images/OpenEnded_mscoco_val2014_questions.json",
        "images/okvqa/images/mscoco_val2014_annotations.json",
    )
