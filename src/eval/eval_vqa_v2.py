"""
Model and data loader for LLaVA evaluation on VQA test data.

VQA dataset: https://visualqa.org/index.html

This code is based on the code written by Haotian Liu in the LLaVA repo at:
(https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/model_vqa_loader.py).
"""

import argparse
import json
import math
import os
from typing import Any

import shortuuid
import torch
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class VQADataset(Dataset):
    """Dataset class for LLaVA evaluation on VQA"""

    def __init__(
        self,
        questions: list[dict],
        image_folder: str,
        conv_mode: str,
        tokenizer: Any,
        image_processor: Any,
        model_config: Any,
    ) -> None:
        """Initialize VQA dataset class

        :param questions: list of JSON question data
        :param image_folder: image folder directory
        :param conv_mode: conversation mode (i.e vicuna_v1)
        :param tokenizer: tokenizer for pretrained model (i.e LLamaTokenizer)
        :param image_processor: image processor for pretrained model (i.e CLIPImageProcessor)
        :param model_config: model config for pretrained model (i.e LLavaConfig)
        """
        self.questions = questions
        self.image_folder = image_folder
        self.split = self.image_folder.split("/")[-1]
        self.conv_mode = conv_mode
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, tuple]:
        """Index into the dataset.

        1. Initialize the prompt with the server message, question, and image token.
        2. Load and preprocess the image.
        3. Tokenize the prompt

        :param index: index in the dataset
        :return input_ids: tokenzed input id tensors
        :return image_tensor: torch tensor of the image
        :return image_size: size of the image
        """
        # Index Question and Image
        sample = self.questions[index]
        qs = sample["question"]
        image_id = str(sample["image_id"]).zfill(12)
        image_filename = f"COCO_{self.split}_{image_id}.jpg"

        # Intialize Prompt ("<Server message>. USER: <image>\n<question>. Assitant:")
        if self.model_config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Preprocess Image
        image = Image.open(os.path.join(self.image_folder, image_filename)).convert("RGB")
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        # Tokenize Prompt
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")

        return input_ids, image_tensor, image.size

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.questions)


def collate_fn(batch: list) -> tuple[torch.Tensor, torch.Tensor, tuple]:
    """Stack batch of data into tensors.

    :param batch: list of data
    :return id_batch: stacked id tensors
    :return image_batch: stacked image tensors
    :return image_size: list of image sizes
    """
    input_ids, image_tensors, image_sizes = zip(*batch)
    id_batch = torch.stack(input_ids, dim=0)
    image_batch = torch.stack(image_tensors, dim=0)
    return id_batch, image_batch, image_sizes


def get_chunk(lst: list[dict], n: int, k: int) -> list[dict]:
    """Get chunk from list.

    :param lst: list to chunk
    :param n: number of chunks
    :param k: chunk to get
    :return: chunk of list
    """
    chunk_size = math.ceil(len(lst) / n)  # integer division
    chunk_list = [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]
    return chunk_list[k]


def create_data_loader(
    questions: list[dict],
    image_folder: str,
    conv_mode: str,
    tokenizer: Any,
    image_processor: Any,
    model_config: Any,
    batch_size: int = 1,
    num_workers: int = 2,  # Colab Restriction
) -> DataLoader:
    """Create the LLaVA VQA dataloader.

    :param questions: list of JSON question data
    :param image_folder: image folder directory
    :param conv_mode: conversation mode (i.e vicuna_v1)
    :param tokenizer: tokenizer for pretrained model (i.e LLamaTokenizer)
    :param image_processor: image processor for pretrained model (i.e CLIPImageProcessor)
    :param model_config: model config for pretrained model (i.e LLavaConfig)
    :param batch_size: batch size for DataLoader
    :param num_workers: number of CPU workers for DataLoader
    :return data_loader: VQA dataloader
    """
    assert batch_size == 1, "batch_size must be 1"
    dataset = VQADataset(
        questions,
        image_folder,
        conv_mode,
        tokenizer,
        image_processor,
        model_config,
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        # num_workers=num_workers, # os.fork() raising Colab error with Multithreaded code
        shuffle=False,
        collate_fn=collate_fn,
    )
    return data_loader


def eval_model(args: argparse.Namespace) -> None:
    """Evaluate LLaVA on VQA test data.

    1. Load pretrained model
    2. Load question and image data and create VQA DataLoader
    3. Generate output answers

    :param args: argparse Namespace args
    """
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, args.model_base, model_name, device_map="auto", offload_folder="offload", offload_state_dict=True
    )

    # Questions
    with open(args.question_file, "r", encoding="utf-8") as f:
        questions = json.load(f)["questions"]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    # Answers Output Directory
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w", encoding="utf-8")  # pylint: disable=consider-using-with

    # DataLoader
    data_loader = create_data_loader(
        questions,
        args.image_folder,
        args.conv_mode,
        tokenizer,
        image_processor,
        model.config,
    )

    # Evaluate
    for (input_ids, image_tensor, image_sizes), sample in tqdm(zip(data_loader, questions), total=len(questions)):
        question_id = sample["question_id"]
        question = sample["question"]

        input_ids = input_ids.to(device="cuda", non_blocking=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True),
                image_sizes=image_sizes,
                do_sample=(args.temperature > 0),
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(
            json.dumps(
                {
                    "question_id": question_id,
                    "prompt": question,
                    "text": outputs,
                    "answer_id": ans_id,
                    "model_id": model_name,
                    "metadata": {},
                }
            )
            + "\n"
        )
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)

    eval_model(parser.parse_args())
