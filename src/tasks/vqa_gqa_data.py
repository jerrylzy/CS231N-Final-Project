# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from param import args
from utils import load_obj_tsv
from .config import USE_MERGED_DATASET

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000

# The path to data and image features.
VQA_DATA_ROOT = 'data/vqa/'
MSCOCO_IMGFEAT_ROOT = 'data/mscoco_imgfeat/'
SPLIT2NAME = {
    'train': 'train2014',
    'valid': 'val2014',
    'minival': 'val2014',
    'nominival': 'val2014',
    'test': 'test2015',
}


class VQAGQADataset:
    """
    A VQA data example in json file:
        {
            "answer_type": "other",
            "img_id": "COCO_train2014_000000458752",
            "label": {
                "net": 1
            },
            "question_id": 458752000,
            "question_type": "what is this",
            "sent": "What is this photo taken looking through?"
        }
        A GQA data example in json file:
        {
            "img_id": "2375429",
            "label": {
                "pipe": 1.0
            },
            "question_id": "07333408",
            "sent": "What is on the white wall?"
        }
    """
    def __init__(self, vqa_splits: str, gqa_splits: str):
        assert vqa_splits != '' or gqa_splits != ''
        self.vqa_name = vqa_splits
        self.gqa_name = gqa_splits

        self.data = []
        gqa_data = []
        if vqa_splits != '':
            self.vqa_splits = vqa_splits.split(',') if vqa_splits is not None else None
            if self.vqa_splits != None:
                # Loading VQA datasets
                for split in self.vqa_splits:
                    self.data.extend(json.load(open("data/vqa/%s.json" % split)))
                print("Load %d VQA data from split(s) %s." % (len(self.data), self.vqa_name))

        # # Convert list to dict (for evaluation)
        # self.vqa_id2datum = {
        #     datum['question_id']: datum for datum in self.data
        # }

        if gqa_splits != '':
            self.gqa_splits = gqa_splits.split(',') if gqa_splits is not None else None
            if self.gqa_splits != None:
                # Loading GQA datasets to data
                for split in self.gqa_splits:
                    gqa_data.extend(json.load(open("data/gqa/%s.json" % split)))
                print("Load %d GQA data from split(s) %s." % (len(gqa_data), self.gqa_name))

        # # Convert list to dict (for evaluation)
        # self.gqa_id2datum = {
        #     datum['question_id']: datum for datum in gqa_data
        # }

        # Merge training dataset
        self.data.extend(gqa_data)

        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['question_id']: datum for datum in self.data
        }

        # Answers (the same answer - label mapping between VQA and GQA)
        ANS2LABEL_PATH = 'data/vqa/trainval_ans2label_merged.json'
        LABEL2ANS_PATH = 'data/vqa/trainval_label2ans_merged.json'

        self.ans2label = json.load(open(ANS2LABEL_PATH))
        self.label2ans = json.load(open(LABEL2ANS_PATH))
        assert len(self.ans2label) == len(self.label2ans)
        for ans, label in self.ans2label.items():
            assert self.label2ans[label] == ans

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)

class GQABufferLoader():
    def __init__(self):
        self.key2data = {}

    def load_data(self, name, number):
        if name == 'testdev':
            path = "data/vg_gqa_imgfeat/gqa_testdev_obj36.tsv"
        else:
            path = "data/vg_gqa_imgfeat/vg_gqa_obj36.tsv"
        key = "%s_%d" % (path, number)
        if key not in self.key2data:
            self.key2data[key] = load_obj_tsv(
                path,
                topk=number
            )
        return self.key2data[key]


gqa_buffer_loader = GQABufferLoader()

"""
An example in obj36 tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
FIELDNAMES would be keys in the dict returned by load_obj_tsv.

Example in obj tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
"""
class VQAGQATorchDataset(Dataset):
    def __init__(self, dataset: VQAGQADataset):
        super().__init__()
        self.raw_dataset = dataset

        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = None

        # Loading detection features to img_data
        img_data = []
        if dataset.vqa_splits != None:
            for split in dataset.vqa_splits:
                # Minival is 5K images in MS COCO, which is used in evaluating VQA/LXMERT-pre-training.
                # It is saved as the top 5K features in val2014_***.tsv
                load_topk = 5000 if (split == 'minival' and topk is None) else topk
                img_data.extend(load_obj_tsv(
                    os.path.join(MSCOCO_IMGFEAT_ROOT, '%s_obj36.tsv' % (SPLIT2NAME[split])),
                    topk=load_topk))

        # Since images in train and valid both come from Visual Genome,
        # buffer the image loading to save memory.
        if dataset.gqa_splits != None:
            if topk == None:
                topk = -1
            if 'testdev' in dataset.gqa_splits or 'testdev_all' in dataset.gqa_splits:     # Always loading all the data in testdev
                img_data.extend(gqa_buffer_loader.load_data('testdev', -1))
            else:
                img_data.extend(gqa_buffer_loader.load_data('train', topk))

        # Convert img list to dict
        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        # Only kept the data with loaded image features
        self.data = []
        for datum in self.raw_dataset.data:
            if datum['img_id'] in self.imgid2img:
                self.data.append(datum)
        print("Use %d data in torch dataset" % (len(self.data)))
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        img_id = datum['img_id']
        ques_id = datum['question_id']
        ques = datum['sent']

        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        feats = img_info['features'].copy()
        boxes = img_info['boxes'].copy()
        assert obj_num == len(boxes) == len(feats)

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        # Provide label (target)
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in label.items():
                if ans in self.raw_dataset.ans2label:
                    target[self.raw_dataset.ans2label[ans]] = score
            return ques_id, feats, boxes, ques, target
        else:
            return ques_id, feats, boxes, ques


class VQAGQAEvaluator:
    def __init__(self, dataset: VQAGQADataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict):
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if ans in label:
                score += label[ans]
        return score / len(quesid2ans)

    def dump_result(self, quesid2ans: dict, path: str, dataset: str):
        """
        Dump results to a json file, which could be submitted to the VQA online evaluation.
        VQA json file submission requirement:
            results = [result]
            result = {
                "question_id": int,
                "answer": str
            }
        
        :param quesid2ans: dict of quesid --> ans
        :param path: The desired path of saved file.
        :param dataset: The desired dataset to use
        """
        is_vqa = dataset == 'vqa'
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                if is_vqa:
                    result.append({
                        'question_id': ques_id,
                        'answer': ans
                    })
                else: # gqa
                  result.append({
                    'questionId': ques_id,
                    'prediction': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)

