import os
from dataclasses import dataclass, field
from typing import  Optional
from transformers import AutoTokenizer, set_seed
from tea import DataCollatorForQA, TEAConfig, TEA_model, ViteVQA_Dataset
from datasets import load_metric
import logging
import transformers
import sys
from transformers.trainer_utils import get_last_checkpoint, is_main_process
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    # init model name or path
    model_name_or_path: Optional[str] = field(default="t5-base")

    # aggregation module
    qformer_module_model_name: str = field(default="Salesforce/instructblip-vicuna-7b",
                                           metadata={"help": "tokenizer init"})
    qm_ques_guide_len: int = field(default=40, metadata={"help": "the length of question instruct in aggregation module"})

    # spatial-temporal module
    am_ocr_multi_gran: bool = field(default=True, metadata= {"help": "OCR multi granularity information provided or not"})
    am_adapter_channels: int = field(default=384, metadata={"help": "channels length"})
    am_adapter_kernel_size_t: int = field(default=3, )
    am_adapter_kernel_size_l: int = field(default=1, )
    am_bbox_scale:int = field(default=1, metadata={"help": "bbox scale"})
    bbox_module_model_name: str = field(default="bros-base-uncased",metadata={"help": "bbox init"})

@dataclass
class DataArguments:
    dataset_name: Optional[str] = field(default='m4-vitevqa', metadata={"help": "The name of the dataset to use (via the datasets library)."})
    vtvqa_taskname: Optional[str] = field(default='t1s1', metadata={"help": "m4-vitvqa dataset split t1s1, t1s2, t2"})
    dataset_root: str = field(default='/data2/zhangyan/data/tea_dataset/vitevqa/', metadata={"help": "dataset root"})
    num_frames: int = field(default=3, )

    # preprocess
    ques_len: int = field(default=45)
    answer_len: int = field(default=25)
    ocr_len: int = field(default=200)
    max_length: int = field(default=65, )
    obj_len: int = field(default=60)

    # module
    use_obj: bool = field(default=False, )
    use_bbox: bool = field(default=False)
    use_feature: bool = field(default=False)
    use_frcn: bool = field(default=False)
    use_frame_type: bool = field(default=False,)
    use_key_frame_module: bool = field(default=False,)
    use_aggregation_module: bool = field(default=False)

    output_attentions: bool = field(default=False,)
    vis_image: bool = field(default=False,)


    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
                    "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
                    "value if set."
        },
    )

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    output_dir: str = field(default="./results/demo")
    do_train: bool = field(default=True)
    do_eval: bool = field(default=True)
    do_predict: bool = field(default=False)

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    cache_dir: Optional[str] = field(default='./cache_dir')
    predict_with_generate: bool = field(default=True)

    report_to: str = field(default="none")
    run_name: str = field(default="demo")
    tags: str =field(default="baseline2")



def train():

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # args
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)


    # model init
    config = TEAConfig.from_pretrained(
        model_args.model_name_or_path,

        # normal settings
        num_frames=data_args.num_frames,
        ques_len=data_args.ques_len,
        ocr_len=data_args.ocr_len,
        obj_len=data_args.obj_len,
        use_bbox=data_args.use_bbox,
        use_frame_type=data_args.use_frame_type,
        max_length=data_args.max_length,

        # aggregation module
        qformer_module_model_name=model_args.qformer_module_model_name,
        use_aggregation_module=data_args.use_aggregation_module,


        # spatial and temporal module
        am_ocr_multi_gran=model_args.am_ocr_multi_gran,
        am_adapter_channels=model_args.am_adapter_channels,
        am_adapter_kernel_size_t=model_args.am_adapter_kernel_size_t,
        am_adapter_kernel_size_l=model_args.am_adapter_kernel_size_l,
        am_bbox_scale=model_args.am_bbox_scale,
        bbox_module_model_name=model_args.bbox_module_model_name,
        #
        output_attentions=data_args.output_attentions,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    processor = None

    model = TEA_model.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        config=config,
    )

    if data_args.dataset_name == "m4-vitevqa":
        dataset_callback = ViteVQA_Dataset
    else:
        raise NotImplementedError()

    if training_args.do_train:
        train_dataset = dataset_callback(data_split="train", tokenizer=tokenizer, processor=processor,
                    data_args=data_args, model_args=model_args, max_samples=data_args.max_train_samples)

    if training_args.do_eval:
        eval_dataset = dataset_callback(data_split="validation", tokenizer=tokenizer, processor=processor,
                    data_args=data_args, model_args=model_args, max_samples=data_args.max_val_samples)

    if training_args.do_predict:
        test_dataset = dataset_callback(data_split="test", tokenizer=tokenizer, processor=processor,
                    data_args=data_args, model_args=model_args, max_samples=data_args.max_test_samples)


    # Data collator
    data_collator = DataCollatorForQA(
        tokenizer,
        model=model,
    )

    anls_metric = load_metric("tea/metric/anls_metric.py")
    stvqa_acc_metric = load_metric("tea/metric/stvqa_acc_metric.py")
    def compute_vqa_metrics(p):
        results = dict()
        predictions, labels, all_answers = p
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        results["anls"] = anls_metric.compute(predictions=decoded_preds, references=all_answers)
        results["stvqa_acc"] = stvqa_acc_metric.compute(predictions=decoded_preds, references=all_answers)

        return results



    # Initialize our Trainer
    from tea import TextVQATrainer
    trainer = TextVQATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_vqa_metrics,
    )

    # Training
    if training_args.do_train:
        checkpoint = last_checkpoint if last_checkpoint else None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        # trainer.save_model()  # Saves the tokenizer too for easy upload

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # test
    if training_args.do_eval:
        logger.info("*** Predict ***")
        predictions, labels, metrics = trainer.predict(eval_dataset, max_length=data_args.max_length)

        print(metrics)
        trainer.log_metrics("validation", metrics)
        trainer.save_metrics("validation", metrics)

    # test
    if training_args.do_predict:
        logger.info("*** Predict ***")
        predictions, labels, metrics = trainer.predict(test_dataset, max_length=data_args.max_length)

        print(metrics)
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)



if __name__ == "__main__":
    train()