"""
Interface for accessing the VQA dataset.

This code is based on the code written by Tsung-Yi Lin for MSCOCO Python API available at the following link:
(https://github.com/pdollar/coco/blob/master/PythonAPI/pycocotools/coco.py) and the VQA code written at
(https://github.com/GT-Vision-Lab/VQA/blob/master/PythonHelperTools/vqaTools/vqa.py)
"""

import json


class VQA:
    """VQA interface module"""

    def __init__(self, annotation_file: str, question_file: str) -> None:
        """VQA Interface intialization

        :param annotation_file: file path to VQA annotation file
        :param question_file: file path the VQA question file
        """
        print("Loading VQA annotations and questions into memory...")
        with open(annotation_file, "r", encoding="utf-8") as f:
            self.dataset: dict = json.load(f)
        with open(question_file, "r", encoding="utf-8") as f:
            self.questions: dict = json.load(f)

        print("Creating index...")
        self.img_to_qa: dict = {ann["image_id"]: [] for ann in self.dataset["annotations"]}
        self.qa: dict = {ann["question_id"]: [] for ann in self.dataset["annotations"]}
        self.qqa: dict = {ann["question_id"]: [] for ann in self.dataset["annotations"]}
        for ann in self.dataset["annotations"]:
            self.img_to_qa[ann["image_id"]] += [ann]
            self.qa[ann["question_id"]] = ann
        for ques in self.questions["questions"]:
            self.qqa[ques["question_id"]] = ques
        print("Index created!")

    def info(self) -> None:
        """Print information about the VQA annotation file."""
        for key, value in self.dataset["info"].items():
            print(f"{key}: {value}")

    # pylint: disable=dangerous-default-value
    def get_ques_ids(
        self, img_ids: list[int] | int = [], ques_types: list[str] | str = [], ans_types: list[str] | str = []
    ) -> list[int]:
        """Get question ids that satisfy given filter conditions. Default skips that filter.

        :param 	img_ids: get question ids for given imgs
        :param ques_types: get question ids for given question types
        :param ans_types: get question ids for given answer types
        :return ids: integer array of question ids
        """
        img_ids = img_ids if isinstance(img_ids, list) else [img_ids]
        ques_types = ques_types if isinstance(ques_types, list) else [ques_types]
        ans_types = ans_types if isinstance(ans_types, list) else [ans_types]

        if len(img_ids) == len(ques_types) == len(ans_types) == 0:
            anns = self.dataset["annotations"]
        else:
            if len(img_ids) != 0:
                anns = sum([self.img_to_qa[img_id] for img_id in img_ids if img_id in self.img_to_qa], [])
            else:
                anns = self.dataset["annotations"]
            anns = anns if len(ques_types) == 0 else [ann for ann in anns if ann["question_type"] in ques_types]
            anns = anns if len(ans_types) == 0 else [ann for ann in anns if ann["answer_type"] in ans_types]
        ids = [ann["question_id"] for ann in anns]
        return ids

    # pylint: disable=dangerous-default-value
    def get_img_ids(
        self, ques_ids: list[int] | int = [], ques_types: list[str] | str = [], ans_types: list[str] | str = []
    ) -> list[int]:
        """Get image ids that satisfy given filter conditions. Default skips that filter.

        :param ques_ids: get image ids for given question ids
        :param ques_types: get image ids for given question types
        :param ans_types: get image ids for given answer types
        :return ids: integer array of image ids
        """
        ques_ids = ques_ids if isinstance(ques_ids, list) else [ques_ids]
        ques_types = ques_types if isinstance(ques_types, list) else [ques_types]
        ans_types = ans_types if isinstance(ans_types, list) else [ans_types]

        assert isinstance(ques_ids, list) and isinstance(ques_types, list) and isinstance(ans_types, list)

        if len(ques_ids) == len(ques_types) == len(ans_types) == 0:
            anns = self.dataset["annotations"]
        else:
            if len(ques_ids) != 0:
                anns = sum([self.qa[ques_id] for ques_id in ques_ids if ques_id in self.qa], [])
            else:
                anns = self.dataset["annotations"]
            anns = anns if len(ques_types) == 0 else [ann for ann in anns if ann["question_type"] in ques_types]
            anns = anns if len(ans_types) == 0 else [ann for ann in anns if ann["answer_type"] in ans_types]
        ids = [ann["image_id"] for ann in anns]
        return ids

    def load_qa(self, ids: list[int]) -> list:
        """Load questions and answers with the specified question ids.

        :param ids: integer ids specifying question ids
        :return qa: loaded qa objects
        """
        return [self.qa[id] for id in ids]

    def show_qa(self, anns: list) -> None:
        """Display the specified annotations.

        :param anns (array of object): annotations to display
        """
        if len(anns) == 0:
            return
        for ann in anns:
            ques_id = ann["question_id"]
            print(f"Question: {self.qqa[ques_id]['question']}")
            for ans in ann["answers"]:
                print(f"Answer {ans['answer_id']}: {ans['answer']}")
