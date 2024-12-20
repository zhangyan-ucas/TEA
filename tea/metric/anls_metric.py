import datasets
import editdistance

_CITATION = \
    """@misc{anls_star,
    title={ANLS* -- A Universal Document Processing Metric for Generative Large Language Models},
    author={David Peer and Philemon Schöpf and Volckmar Nebendahl and Alexander Rietzler and Sebastian Stabinger},
    year={2024},
    eprint={2402.03848},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}"""

class ANLS_metric(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description="VQA ANLS",
            inputs_description="preds, references require list", # to do
            citation=_CITATION,
            features=datasets.Features({
                    "predictions": datasets.Value("string"),
                    "references": datasets.Sequence(datasets.Value("string")),
            })
        )

    def get_anls(self, s1, s2):
        s1 = s1.lower().strip()
        s2 = s2.lower().strip()
        iou = 1 - editdistance.eval(s1, s2) / max(len(s1), len(s2))
        anls = iou if iou >= .5 else 0.
        return anls



    def _compute(self, predictions, references):
        pred_scores = []
        for pred, answer in zip(predictions, references):
            anls = max(
                self.get_anls(pred, gt)
                for gt in answer
            )
            pred_scores.append(anls)

        accuracy = sum(pred_scores) / len(pred_scores)

        return accuracy


if __name__ == "__main__":
    metric = ANLS_metric()
    metric._compute(predictions=["123", "abcd"], references= [["123"]* 10, ["abcd"] * 10])