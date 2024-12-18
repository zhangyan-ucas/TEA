
<h1 align="center"> TEA: Extending TextVQA from Image to Video with Spatio-Temporal Clues</h1>
<p align="center">
<h4 align="center">This is the official repository of the paper <a href="https://arxiv.org/abs/2412.12502">TEA</a>.</h4>
<h5 align="center"><em><a href="https://scholar.google.com/citations?hl=zh-CN&user=IUNcUO0AAAAJ">Yan Zhang</a>, Gangyan Zeng, Huawen Shen, Daiqing Wu, Yu Zhou, Can Ma </em></h5>
<p align="center">
  <a href="#introduction">Introduction</a> |
  <a href="#news">News</a> |
  <a href="#usage">Usage</a> |
  <a href="#main results">Main Results</a> |
  <a href="#statement">Statement</a>
</p>


# News
***2024/12/12***
- Update the training codes and M4-ViteVQA datasets features!

***2024/12/10***
- The paper is accepted to 2025 AAAI! 


### Installation
Python_3.8 + PyTorch_2.1.0 + CUDA_12.1 + transformers_4.42.3

```python
cd TEA-main
conda create -n tea python=3.8 -y
conda activate tea
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Datasets prepare

Following features are provided for [M4-ViteVQA](https://github.com/bytedance/VTVQA) dataset:
- Raw video frames
- OCR tokens and multi-granularity layout (Extracted by [TransVTSpotter](https://github.com/bytedance/VTVQA), [ABINet](https://github.com/bytedance/VTVQA) and [Hi-SAM](https://github.com/ymy-k/Hi-SAM), respectively)
- OBJ tokens (Extracted by Faster R-CNN)


Annotations | hier_ocr | img_dir | video_ocr



```
|- ./tea_dataset
    |--- vitevqa
    |      |--- Annotations (inclueds annos for M4-ViteVQA)
    |            |--- ViteVQA_0.0.2_t1s1train.json
    |            |--- ViteVQA_0.0.2_t1s1val.json
    |            |--- ViteVQA_0.0.2_t1s1test.json
    |            |--- ...
    |      |--- hier_ocr (inclueds multi-granularity layout)
    |            |--- 02502 (video_name)
    |            |--- 02915 (video_name)
    |            |--- ...
    |      |--- img_dir (inclueds raw video frames)
    |            |--- 02502 (video_name)
    |            |--- 02915 (video_name)
    |            |--- ...
    |      |--- video_ocr (inclueds video text ocr results)
    |            |--- 02502.json (video_name)
    |            |--- 02915.json (video_name)
    |            |--- ...
```

### 

## Ckpt Download




## Training
### M4-ViteVQA

```
sh ./scripts/vitevqa.sh
```



# Statement

This project is for research purpose only. For any other questions please contact [zhangyan2022@iie.ac.cn](mailto:zhangyan2022@iie.ac.cn).
