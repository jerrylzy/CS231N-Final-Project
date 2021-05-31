import json

VQA_PATH = '../vqa/trainval_label2ans.json'
GQA_PATH = '../gqa/trainval_label2ans.json'

VQA_LABEL2ANS_PATH = '../vqa/trainval_label2ans_merged.json'
GQA_LABEL2ANS_PATH = '../gqa/trainval_label2ans_merged.json'

VQA_ANS2LABEL_PATH = '../vqa/trainval_ans2label_merged.json'
GQA_ANS2LABEL_PATH = '../gqa/trainval_ans2label_merged.json'

def main():
    with open(VQA_PATH) as f:
        vqa_labels = json.load(f)

    with open(GQA_PATH) as f:
        gqa_labels = json.load(f)

    vqa_labels_set = set(vqa_labels)
    gqa_labels_set = set(gqa_labels)
    merged_labels = list(vqa_labels_set.union(gqa_labels_set))

    with open(VQA_LABEL2ANS_PATH, 'w') as f:
        json.dump(merged_labels, f)

    with open(GQA_LABEL2ANS_PATH, 'w') as f:
        json.dump(merged_labels, f)

    as2label_map_merged = dict()

    counter = 0
    for label in merged_labels:
        as2label_map_merged[label] = counter
        counter += 1

    with open(VQA_ANS2LABEL_PATH, 'w') as f:
        json.dump(as2label_map_merged, f)

    with open(GQA_ANS2LABEL_PATH, 'w') as f:
        json.dump(as2label_map_merged, f)
    




if __name__ == '__main__':
    main()