from torch.utils.data import Dataset
import transformers
import json
import numpy as np
import torch
import os
import random
from transformers import AutoTokenizer, AutoProcessor
from PIL import Image
from typing import Optional
import torchvision



class ViteVQA_Dataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_split: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 model_args: None,
                 data_args: None,
                 processor: Optional[transformers.DataProcessor] = None,
                 max_samples: Optional[int] = None):
        super(ViteVQA_Dataset, self).__init__()
        self.data_split = data_split
        dataset_root = data_args.dataset_root 

        if data_args.vtvqa_taskname == "t1s1":
            self.datasets_dict = {
                "train": {
                    "image_dir": f"{dataset_root}img_dir",
                    "annotations": f"{dataset_root}Annotations/ViteVQA_0.0.2_t1s1train.json",
                    "ocr_dir": f"{dataset_root}video_ocr",
                    "multi_ocr_bbox_dir": f"{dataset_root}hier_ocr",
                    "obj_dir":f"{dataset_root}OBJ",
                },
                "validation": {
                    "image_dir": f"{dataset_root}img_dir",
                    "annotations": f"{dataset_root}Annotations/ViteVQA_0.0.2_t1s1val.json",
                    "ocr_dir": f"{dataset_root}video_ocr",
                    "multi_ocr_bbox_dir": f"{dataset_root}hier_ocr",
                    "obj_dir": f"{dataset_root}OBJ",

                },
                "test": {
                    "image_dir": f"{dataset_root}img_dir",
                    "annotations": f"{dataset_root}Annotations/ViteVQA_0.0.2_t1s1test.json",
                    "ocr_dir": f"{dataset_root}video_ocr",
                    "multi_ocr_bbox_dir": f"{dataset_root}hier_ocr",
                    "obj_dir": f"{dataset_root}OBJ",

                }
            }
        elif data_args.vtvqa_taskname == "t1s2":
            self.datasets_dict = {
                "train": {
                    "image_dir": f"{dataset_root}img_dir",
                    "annotations": f"{dataset_root}Annotations/ViteVQA_0.0.2_t1s2train.json",
                    "ocr_dir": f"{dataset_root}video_ocr",
                    "multi_ocr_bbox_dir": f"{dataset_root}hier_ocr",
                    "obj_dir": f"{dataset_root}OBJ",

                },
                "validation": {
                    "image_dir": f"{dataset_root}img_dir",
                    "annotations": f"{dataset_root}Annotations/ViteVQA_0.0.2_t1s2val.json",
                    "ocr_dir": f"{dataset_root}video_ocr",
                    "multi_ocr_bbox_dir": f"{dataset_root}hier_ocr",
                    "obj_dir": f"{dataset_root}OBJ",

                },
                "test": {
                    "image_dir":  f"{dataset_root}img_dir",
                    "annotations": f"{dataset_root}Annotations/ViteVQA_0.0.2_t1s2test.json",
                    "ocr_dir": f"{dataset_root}video_ocr",
                    "multi_ocr_bbox_dir": f"{dataset_root}hier_ocr",
                    "obj_dir": f"{dataset_root}OBJ",

                }
            }
        elif data_args.vtvqa_taskname == "t2":
            self.datasets_dict = {
                "train": {
                    "image_dir": f"{dataset_root}img_dir",
                    "annotations": f"{dataset_root}Annotations/ViteVQA_0.0.2_t2train.json",
                    "ocr_dir": f"{dataset_root}video_ocr",
                    "multi_ocr_bbox_dir": f"{dataset_root}hier_ocr",
                    "obj_dir": f"{dataset_root}OBJ",

                },
                "validation": {
                    "image_dir": f"{dataset_root}img_dir",
                    "annotations": f"{dataset_root}Annotations/ViteVQA_0.0.2_t2val.json",
                    "ocr_dir": f"{dataset_root}video_ocr",
                    "multi_ocr_bbox_dir": f"{dataset_root}hier_ocr",
                    "obj_dir": f"{dataset_root}OBJ",

                },
                "test": {
                    "image_dir": f"{dataset_root}img_dir",
                    "annotations": f"{dataset_root}Annotations/ViteVQA_0.0.2_t2test.json",
                    "ocr_dir": f"{dataset_root}video_ocr",
                    "multi_ocr_bbox_dir": f"{dataset_root}hier_ocr",
                    "obj_dir": f"{dataset_root}OBJ",
                }
            }

        else: raise NotImplementedError

        self.image_dir = self.datasets_dict[self.data_split]["image_dir"]
        self.ocr_dir = self.datasets_dict[self.data_split]["ocr_dir"]

        self.obj_vocab = {idx:i.replace("\n", "") for idx, i in enumerate(open(f"{dataset_root}objects_vocab.txt", encoding="utf-8", mode="r").readlines())}
        self.obj_dir = self.datasets_dict[self.data_split]["obj_dir"]

        # config
        self.tokenizer = tokenizer
        self.processor = processor
        self.data_args = data_args
        self.model_args = model_args
        self.num_frames = self.data_args.num_frames
        self.max_samples = max_samples

        if self.data_args.use_aggregation_module :
            self.qm_tokenizer = AutoTokenizer.from_pretrained(self.model_args.qformer_module_model_name + "/qformer_tokenizer")
            self.vision_processor = AutoProcessor.from_pretrained(self.model_args.qformer_module_model_name)

        # init annotation
        self.init_annotations()

    def init_annotations(self):
        self.annotations = json.load(open(self.datasets_dict[self.data_split]["annotations"],
                                          encoding="utf-8", mode="r"))['data']

        if self.max_samples: self.annotations = random.sample(self.annotations, self.max_samples)


    def __len__(self):
        return len(self.annotations)


    def normalized_bbox(self, bbox, size):
        """
        :param bbox:
        :param size: weight, height
        :return:
        """

        x_min, x_max = min(bbox[::2])/size[0], max(bbox[::2])/size[0]
        y_min, y_max = min(bbox[1::2]) / size[1], max(bbox[1::2]) / size[1]


        return [x_min, y_min, x_max, y_max]

    def select_frame_idx(self, frame_idx_list, num_frames):
        """
        :param frame_idx_list:
        :param num_frames:
        :return:
        """
        if num_frames == 1:
            select_frame_idx = [frame_idx_list[len(frame_idx_list)//2]]
        else:
            interval = (len(frame_idx_list) - 1) // (num_frames - 1)
            select_frame_idx = list(range(0, interval * (num_frames - 1) + 1, interval))
            select_frame_idx = [frame_idx_list[i] for i in select_frame_idx]
        return select_frame_idx

    def eight2four(self, bbox):
        x_min, x_max = min(bbox[::2]) , max(bbox[::2])
        y_min, y_max = min(bbox[1::2]), max(bbox[1::2])

        return [x_min, y_min, x_max, y_max]


    def match2linepara(self, bbox, line_bbox, para_bbox, size):

        line_idx = torch.max(torchvision.ops.box_iou(torch.tensor(bbox), torch.tensor(line_bbox)), dim=1)[1]
        para_idx = torch.max(torchvision.ops.box_iou(torch.tensor(bbox), torch.tensor(para_bbox)), dim=1)[1]

        norm_line_bbox = [self.normalized_bbox(line_bbox[i], size) for i in line_idx.tolist()]
        norm_para_bbox = [self.normalized_bbox(para_bbox[i], size) for i in para_idx.tolist()]
        return norm_line_bbox, norm_para_bbox

    def __getitem__(self, guid):
        # init
        random.seed(42)
        sample = self.annotations[guid]
        
        
        ########## init ocr ##########
        image_paths = []
        ocr_tokens, ocr_bbox, ocr_frcn = [], [], []
        obj_tokens, obj_bbox, obj_frcn = [], [], []
        if self.model_args.am_ocr_multi_gran:
            line_ocr_bbox, para_ocr_bbox = [], []

        size = [sample['video_width'], sample['video_height']]

        ocr_data = json.load(open(f"{self.ocr_dir}/{sample['video_id']}.json", encoding="utf-8", mode="r"))

        # iter ocr results
        if len(ocr_data) == 0:
            pass

        elif self.num_frames > 0:
            exist_ocr_frame_idx = [i for i,j in ocr_data.items() if len(j) != 0]
            if len(exist_ocr_frame_idx) >= self.num_frames:
                select_frame_idx = self.select_frame_idx(exist_ocr_frame_idx, self.num_frames)
            else:
                select_frame_idx = [i for i in ocr_data.keys() if i not in exist_ocr_frame_idx]
                select_frame_idx = self.select_frame_idx(select_frame_idx, self.num_frames - len(exist_ocr_frame_idx))
                select_frame_idx = sorted(select_frame_idx + exist_ocr_frame_idx)

            assert len(select_frame_idx) == self.num_frames
            image_paths = [os.path.join(self.image_dir, sample["video_id"], i + ".jpg") for i in select_frame_idx]


            for idx, frame_idx in enumerate(select_frame_idx):
                # ocr tokens
                frame_ocr_data = ocr_data[str(frame_idx)]

                if self.model_args.am_ocr_multi_gran:
                    multi_ocr_bbox_path = os.path.join(self.datasets_dict[self.data_split]["multi_ocr_bbox_dir"], sample['video_id'], frame_idx + ".jsonl")
                    multi_ocr_bbox_data = json.load(open(multi_ocr_bbox_path, "r"))
                    bbox = [self.eight2four(i['points']) for i in frame_ocr_data]

                    if len(bbox) == 0:
                        line_ocr_bbox.append([[0.0, 0.0, 0.0, 0.0]])
                        para_ocr_bbox.append([[0.0, 0.0, 0.0, 0.0]])

                    elif "line_bbox" not in multi_ocr_bbox_data or len(multi_ocr_bbox_data["line_bbox"]) == 0:
                        line_ocr_bbox.append(
                            [[0.0, 0.0, 0.0, 0.0]] +
                            [self.normalized_bbox(i['points'], size) for i in frame_ocr_data]
                        )
                        para_ocr_bbox.append(
                            [[0.0, 0.0, 0.0, 0.0]] +
                            [self.normalized_bbox(i['points'], size) for i in frame_ocr_data]
                        )

                    else:
                        frame_line_ocr_bbox, frame_para_ocr_bbox = multi_ocr_bbox_data["line_bbox"], multi_ocr_bbox_data["para_bbox"]
                        frame_line_ocr_bbox, frame_para_ocr_bbox = self.match2linepara(
                            bbox=bbox,
                            line_bbox=frame_line_ocr_bbox,
                            para_bbox=frame_para_ocr_bbox,
                            size=size
                        )

                        line_ocr_bbox.append([[0.0, 0.0, 0.0, 0.0]] + frame_line_ocr_bbox)
                        para_ocr_bbox.append([[0.0, 0.0, 0.0, 0.0]] + frame_para_ocr_bbox)


                ocr_tokens.append(["context: "] + [i['ocr'] for i in frame_ocr_data])

                ocr_bbox.append(
                    [[0.0, 0.0, 0.0, 0.0]] +
                    [self.normalized_bbox(i['points'], size) for i in frame_ocr_data]
                )

                # obj_tokens
                if self.data_args.use_obj:

                    frame_obj_data = np.load(os.path.join(self.obj_dir, f"{sample['video_id']}/{frame_idx}.npy"), allow_pickle=True).tolist()
                    obj_tokens.append(
                        ["context: "] + [self.obj_vocab[i] for i in frame_obj_data['objects_id'].tolist()])
                    obj_bbox.append([[0.0, 0.0, 0.0, 0.0]] + [self.normalized_bbox(i.tolist(), size) for i in frame_obj_data['boxes']])
                    
        

        # init
        model_inputs = {}

        # preprocess for encoder inputs
        # preprocess for ocr tokens
        ocr_tokenized = self.tokenizer(ocr_tokens, max_length=self.data_args.ocr_len, truncation=True,
                                    padding="max_length", is_split_into_words=True, return_tensors="pt")
        model_inputs["input_ids"] = ocr_tokenized["input_ids"]
        model_inputs["attention_mask"] = ocr_tokenized["attention_mask"]
        model_inputs["frame_type"] = torch.LongTensor([[i]* self.data_args.ocr_len for i in range(self.data_args.num_frames)])

        sample_ocr_bbox, sample_ocr_frcn = [], []
        if self.model_args.am_ocr_multi_gran: sample_ocr_line_bbox, sample_ocr_para_bbox = [], []
        for frame_idx in range(self.data_args.num_frames):

            sample_ocr_bbox.append(
                [[0.0, 0.0, 0.0, 0.0] if word_idx is None else ocr_bbox[frame_idx][word_idx]
                 for word_idx in ocr_tokenized.word_ids(batch_index=frame_idx)]
            )
            if self.model_args.am_ocr_multi_gran:
                sample_ocr_line_bbox.append(
                    [[0.0, 0.0, 0.0, 0.0] if word_idx is None else line_ocr_bbox[frame_idx][word_idx]
                     for word_idx in ocr_tokenized.word_ids(batch_index=frame_idx)]
                )
                sample_ocr_para_bbox.append(
                    [[0.0, 0.0, 0.0, 0.0] if word_idx is None else para_ocr_bbox[frame_idx][word_idx]
                     for word_idx in ocr_tokenized.word_ids(batch_index=frame_idx)]
                )


        model_inputs["ocr_bbox"] = torch.tensor(sample_ocr_bbox)
        if self.model_args.am_ocr_multi_gran:
            model_inputs["line_ocr_bbox"] = torch.tensor(sample_ocr_line_bbox)
            model_inputs["para_ocr_bbox"] = torch.tensor(sample_ocr_para_bbox)


        # preprocess for obj tokens
        if self.data_args.use_obj:
            obj_tokenized = self.tokenizer(obj_tokens, max_length=self.data_args.obj_len, truncation=True,
                                           padding="max_length", is_split_into_words=True, return_tensors="pt")
            model_inputs["input_ids"] = torch.cat((model_inputs["input_ids"], obj_tokenized["input_ids"]), dim=1)
            model_inputs["attention_mask"] = torch.cat((model_inputs["attention_mask"], obj_tokenized["attention_mask"]), dim=1)
            model_inputs["frame_type"] = torch.cat((model_inputs["frame_type"],
                                    torch.LongTensor([[i]* self.data_args.obj_len for i in range(self.data_args.num_frames)])),dim=1)

            sample_obj_bbox, sample_obj_frcn = [], []
            for frame_idx in range(self.data_args.num_frames):
                sample_obj_bbox.append(
                    [[0.0, 0.0, 0.0, 0.0] if word_idx is None else obj_bbox[frame_idx][word_idx]
                     for word_idx in obj_tokenized.word_ids(batch_index=frame_idx)]
                )
                if self.data_args.use_frcn:
                    sample_obj_frcn.append(
                        [np.zeros((2048, )) if word_idx is None else obj_frcn[frame_idx][word_idx]
                        for word_idx in obj_tokenized.word_ids(batch_index=frame_idx)]
                    )
            model_inputs["ocr_bbox"] = torch.cat((model_inputs["ocr_bbox"], torch.tensor(sample_obj_bbox)), dim=1)

        # preprocess for decoder inputs
        ques_tokenized = self.tokenizer(f'question: {sample["question"]}')["input_ids"]
        answer_tokenized = self.tokenizer(random.choice(sample["answers"]))["input_ids"]

        eval_decoder_input_ids = [0] + ques_tokenized[:-1]
        decoder_input_ids = eval_decoder_input_ids + answer_tokenized
        decoder_input_ids += [0] * (self.data_args.max_length - len(decoder_input_ids))

        labels = [-100] * (len(eval_decoder_input_ids) - 1) + answer_tokenized
        labels += [-100] * (self.data_args.max_length - len(labels))

        model_inputs["eval_decoder_input_ids_idx"] = len(eval_decoder_input_ids)
        model_inputs["decoder_input_ids"] = decoder_input_ids
        model_inputs["labels"] = labels

        # answer
        model_inputs["answers"] = sample["answers"]

        # preprocess for aggregation module
        if self.data_args.use_aggregation_module :
            ques_guide = self.qm_tokenizer(f'question: {sample["question"]}',
                padding="max_length", max_length=self.model_args.qm_ques_guide_len,truncation=True)
            model_inputs["qformer_input_ids"] = ques_guide["input_ids"]
            model_inputs["qformer_attention_mask"] = ques_guide["attention_mask"]

            # add pixel values for CLIP
            model_inputs["pixel_values"] = self.vision_processor(images=[Image.open(i) for i in image_paths],return_tensors="pt")["pixel_values"]


        return model_inputs




