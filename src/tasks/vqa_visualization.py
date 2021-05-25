# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections
import random

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from param import args
from pretrain.qa_answer_table import load_lxmert_qa
from tasks.vqa_model import VQAModel
from tasks.vqa_atten_model import VQAModelAttn
from tasks.vqa_data import VQADataset, VQATorchDataset, VQAEvaluator
import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')


def get_data_tuple(splits: str, bs: int, shuffle=False, drop_last=False) -> DataTuple:
    dset = VQADataset(splits)
    tset = VQATorchDataset(dset)
    evaluator = VQAEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class VQA:
    def __init__(self):
        # Datasets
        self.train_tuple = get_data_tuple(
            args.train, bs=args.batch_size, shuffle=True, drop_last=True
        )
        if args.valid != "":
            self.valid_tuple = get_data_tuple(
                args.valid, bs=1024,
                shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None

        # Model
        self.model = VQAModel(self.train_tuple.dataset.num_answers)

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=self.train_tuple.dataset.label2ans)

        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        # Loss and Optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("BertAdam Total Iters: %d" % t_total)
            from lxrt.optimization import BertAdam
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(self.model.parameters(), args.lr)

        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        best_valid = 0.
        for epoch in range(args.epochs):
            quesid2ans = {}
            for i, (ques_id, feats, boxes, sent, target, _, _) in iter_wrapper(enumerate(loader)):

                self.model.train()
                self.optim.zero_grad()

                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                logit = self.model(feats, boxes, sent)
                assert logit.dim() == target.dim() == 2
                loss = self.bce_loss(logit, target)
                loss = loss * logit.size(1)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid.item()] = ans

            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)

            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")

                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)

            print(log_str, end='')

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

        self.save("LAST")

    def predict(self, eval_tuple: DataTuple, dump=None):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent = datum_tuple[:4]  # Avoid seeing ground truth
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(feats, boxes, sent)
                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid.item()] = ans
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return quesid2ans

    def plot_confidence(self, eval_tuple: DataTuple, dump=None):
        # plot confidence bar graph for one example
        self.model.eval()
        dset, loader, evaluator = eval_tuple

        # sample = random.randint(0, len(loader) - 1)
        sample = 250

        for i, datum_tuple in enumerate(loader):
            if i == sample:
                ques_id, feats, boxes, sent, _, img_id, original_boxes = datum_tuple
                with torch.no_grad():
                    print('image id: ', img_id[0])
                    print('question id: ', ques_id[0])

                    original_boxes = original_boxes[0][1].cpu().numpy()

                    im = Image.open('COCO_val2014_000000572477.jpg')
                    # Create figure and axes
                    fig, ax = plt.subplots()

                    # Display the image
                    ax.imshow(im)

                    # Create a Rectangle patch
                    rect = patches.Rectangle((original_boxes[0],original_boxes[1]), original_boxes[2]-original_boxes[0], original_boxes[3]-original_boxes[1], linewidth=1, edgecolor='r', facecolor='none')

                    # Add the patch to the Axes
                    ax.add_patch(rect)

                    plt.savefig('bbCOCO_val2014_000000572477.jpg')

                    feats, boxes = feats.cuda(), boxes.cuda()
                    logit = self.model(feats, boxes, sent)
                    print(logit)
                    logit = nn.Softmax(dim=1)(logit)

                    for j in range(5):
                        attn_wgts = torch.load('attn_wgts_{}.pt'.format(j))
                        attn_wgts = attn_wgts[0][1:10].cpu().numpy()
                        fig = go.Figure(data=go.Heatmap(
                            z=attn_wgts,
                            # x=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
                            # y=['Morning', 'Afternoon', 'Evening'],
                            ))
                        fig.write_image('atten_vis_{}.png'.format(j))
                        fig.show()

                    scores, labels = torch.topk(logit, 5, dim=1)
                    print(scores)
                    print(labels)
                    answers = []
                    scores = scores[0]
                    labels = labels[0]
                    scores = scores.cpu().numpy() * 100
                    for label in labels.cpu().numpy():
                        answers.append(dset.label2ans[label])

                    fig = go.Figure(data=[go.Bar(
                        x=scores, y=answers,
                        text=scores,
                        textposition='auto',
                        orientation='h',
                        marker=dict(color='lightsalmon')
                    )])
                    fig.update_traces(texttemplate='%{text:.3s}')
                    fig.update_layout(
                        title='Predicted confidence of top-5 answers',
                        yaxis_title='Answers',
                        xaxis_title='Confidence'
                    )
                    fig.write_image('SampleQuestionConfidence.png')

                    # plt.bar(answers, scores)
                    # plt.xlabel('Answers')
                    # plt.ylabel('Confidence')
                    # plt.title('Predicted confidence of top-5 answers')
                    # plt.savefig('SampleQuestionConfidence.png', format='png')

                break

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        """Evaluate all data in data_tuple."""
        quesid2ans = self.predict(eval_tuple, dump)
        return eval_tuple.evaluator.evaluate(quesid2ans)

    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, (ques_id, feats, boxes, sent, target) in enumerate(loader):
            _, label = target.max(1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                ans = dset.label2ans[l]
                quesid2ans[qid.item()] = ans
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)


if __name__ == "__main__":
    # Build Class
    vqa = VQA()

    # Load VQA model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    if args.load is not None:
        vqa.load(args.load)

    # Test or Train
    if args.test is not None:
        args.fast = args.tiny = False  # Always loading all data in test
        if 'test' in args.test:
            vqa.predict(
                get_data_tuple(args.test, bs=950,
                               shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'test_predict.json')
            )
        elif 'val' in args.test:
            # Since part of valididation data are used in pre-training/fine-tuning,
            # only validate on the minival set.
            # create bar graph for top answers
            vqa.plot_confidence(
                get_data_tuple('minival', bs=1,
                               shuffle=True, drop_last=False),
                dump=os.path.join(args.output, 'minival_predict.json')
            )
        else:
            assert False, "No such test option for %s" % args.test
    else:
        print('Splits in Train data:', vqa.train_tuple.dataset.splits)
        if vqa.valid_tuple is not None:
            print('Splits in Valid data:', vqa.valid_tuple.dataset.splits)
            print("Valid Oracle: %0.2f" % (vqa.oracle_score(vqa.valid_tuple) * 100))
        else:
            print("DO NOT USE VALIDATION")
        vqa.train(vqa.train_tuple, vqa.valid_tuple)


