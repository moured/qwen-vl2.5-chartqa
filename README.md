<!-- ──────────────────────────────────────────────────────────────── -->
<h1 align="center">
  📊&nbsp;&nbsp;<strong>Qwen-VL&nbsp;2.5 Demo&nbsp;Inference&nbsp;for&nbsp;VQA<br/>(example&nbsp;with&nbsp;ChartQA)</strong>&nbsp;🚀
</h1>

<p align="center">
  <a href="https://github.com/QwenLM/Qwen2.5-VL/tree/main">
    <img src="https://img.shields.io/badge/Dataset-ChartQA-11bbff?logo=data&logoColor=white" alt="ChartQA"/>  </a>
  &nbsp;
  <a href="https://github.com/vis-nlp/ChartQA">
    <img src="https://img.shields.io/badge/Model-Qwen2.5-11bbff?logo=data&logoColor=white" alt="ChartQA"/>
  </a>
  &nbsp;
  <a href="./output/evaluation.json">
    <img src="https://img.shields.io/badge/Results-7b-11bbff?logo=data&logoColor=white" alt="ChartQA"/>
  </a>
</p>

## ✨ Introduction
We experimented with the **Qwen-VL 2.5** model series on the **ChartQA** visual-question-answering benchmark. The demo code provided is an inference example with evaluation (relaxed-accuracy) for the VQA benchmark. The code can be tailored to *any* VQA dataset with only minor tweaks—feel free to reach out for guidance!


## 🛠️ How to use

### 1 · Install requirements
```bash
pip install -r requirements.txt
```
```bash
python demo.py \
  --test_human      path/to/chartqa/test_human.json \
  --test_augmented  path/to/chartqa/test_augmented.json \
  --img_root        path/to/chartqa/test/images/folder \
  --out             path/to/output/predictions.json \
```

## 📈 Results

| Split            | Accuracy (%) |
| ---------------- | -----------: |
| `test_human`     |    **80.72** |
| `test_augmented` |    **94.96** |
| **Overall**      |    **87.84** |

(Model: Qwen 2.5-VL-7B-Instruct, FP16 on one A100 80 GB)


## 🤝 Contact
💼 <a href="https://www.linkedin.com/in/omar-moured/">LinkedIn</a>

✉️ moured.omar@gmail.com

## 📚 Citation

```cite
@article{qwen2.5,
    title   = {Qwen2.5 Technical Report}, 
    author  = {An Yang and Baosong Yang and Beichen Zhang and Binyuan Hui and Bo Zheng and Bowen Yu and Chengyuan Li and Dayiheng Liu and Fei Huang and Haoran Wei and Huan Lin and Jian Yang and Jianhong Tu and Jianwei Zhang and Jianxin Yang and Jiaxi Yang and Jingren Zhou and Junyang Lin and Kai Dang and Keming Lu and Keqin Bao and Kexin Yang and Le Yu and Mei Li and Mingfeng Xue and Pei Zhang and Qin Zhu and Rui Men and Runji Lin and Tianhao Li and Tingyu Xia and Xingzhang Ren and Xuancheng Ren and Yang Fan and Yang Su and Yichang Zhang and Yu Wan and Yuqiong Liu and Zeyu Cui and Zhenru Zhang and Zihan Qiu},
    journal = {arXiv preprint arXiv:2412.15115},
    year    = {2024}
}
```

```cite
@inproceedings{masry-etal-2022-chartqa,
    title = "{C}hart{QA}: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning",
    author = "Masry, Ahmed  and
      Long, Do  and
      Tan, Jia Qing  and
      Joty, Shafiq  and
      Hoque, Enamul",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2022",
    month = may,
    year = "2022",
}
```
