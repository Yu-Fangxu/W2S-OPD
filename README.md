<h1 align="center">
  W2S-OPD: Weak-to-Strong On-Policy Distillation
</h1>

<p align="center">
  📄 <a href="https://w2s-opd.github.io"><strong>Paper</strong></a> |
  🌐 <a href="https://w2s-opd.github.io"><strong>Project Page</strong></a>
</p>

<p align="center">
  <a href="https://yu-fangxu.github.io/">Fangxu Yu</a><sup>1</sup>,
  <a href="https://zinanlin.me/">Zinan Lin</a><sup>2</sup>,
  <a href="https://xiaodongagi.github.io/">Xiaodong Liu</a><sup>2</sup>,
  <a href="https://weijia-xu.github.io/">Weijia Xu</a><sup>2</sup>,
  <a href="https://www.microsoft.com/en-us/research/people/michaelxu/">Michael Xu</a><sup>2</sup>,
  <a href="https://tianyizhou.github.io/">Tianyi Zhou</a><sup>3</sup>,
  <a href="https://www.microsoft.com/en-us/research/people/jfgao/">Jianfeng Gao</a><sup>2</sup>
</p>

<p align="center">
  <sup>1</sup>University of Maryland, College Park &nbsp;
  <sup>2</sup>Microsoft Research &nbsp;
  <sup>3</sup>Mohamed Bin Zayed University of Artificial Intelligence
</p>

<img src="assets/main_arch.png" width="100%" />

## 🔥 Overview

**W2S-OPD** improves a **strong student** by distilling it from **multiple weak models**, none of
which is as capable as the student overall. On-policy distillation (OPD) delivers dense, on-policy
token-level supervision, but presupposes a teacher at least as capable as the student — which fails
at the frontier (no larger teacher exists) or requires costly experts trained at the student's scale.

W2S-OPD removes that premise. Given a **stronger positive** model and a **weaker negative** model, their
**logit difference** isolates the *capability direction* that separates them. Adding this direction onto
the student's own base model yields a **proxy teacher** that couples the capability while staying
**distributionally adjacent** to the student. The student distills it by minimizing the per-position
reverse KL on its own rollouts.

```
z_proxy = z_student_base  +  α · ( z_positive − z_negative )      # composed in logit space
teacher = softmax(z_proxy)
```

## 🚀 Key Features

- 🧩 **Proxy teacher in logit space**: build a teacher stronger than any single weak model by adding a
  contrast direction `α·(z₊ − z₋)` onto the student's base — no teacher larger than the student needed.
- 🎛️ **Three interchangeable contrasts**: post-RL expert vs. its init, larger vs. smaller base, or a base
  model under correct vs. incorrect hints — swap via one env var, everything else fixed.
- 🔌 **Drop-in on [verl](https://github.com/volcengine/verl)**: FSDP + vLLM rollout, single/multi-node;
  only the teacher's per-position top-K distribution is exported to the student.

## 🖥️ Installation

```bash
git clone https://github.com/Yu-Fangxu/W2S-OPD.git && cd W2S-OPD
conda create -n w2s-opd python=3.10 -y && conda activate w2s-opd
pip install -e verl
pip install -r verl/requirements-cuda.txt
pip install math_verify evalplus
```

## 📥 Data & Models

Training data (verl parquet with `prompt`, `reward_model.ground_truth`, `extra_info`):
- **Math**: DeepMath-103K → `data/DeepMath-103K/train.parquet`
- **Code**: Eurus / TACO → `data/Eurus/code_train.parquet`

Models are loaded from Hugging Face — set the paths inside each `scripts/setting*.sh`. The student is
`Qwen3-8B`; the positive / negative come from the Qwen3 family (`Qwen3-4B-Base`, post-RL 4B experts,
`Qwen3-1.7B`, `Qwen3-0.6B`).

## 🎯 Training

Each contrast is one script (edit the model paths at the top, then run):

| Setting | Positive `z₊` | Negative `z₋` | Isolates |
|---|---|---|---|
| `scripts/setting1_rl_skill_delta.sh` | post-RL expert | its pre-RL initialization | the skill RL instills |
| `scripts/setting2_capacity_delta.sh` | larger base model | smaller base model | capability from scale |
| `scripts/setting3_context_delta.sh`  | base + correct hint | base + incorrect hint | instance-level direction to the solution |

```bash
bash scripts/setting1_rl_skill_delta.sh
```

Common knobs (env → `verl/examples/g_opd/run_topk_opd_8b.sh`):
`PROXY_ALPHA` (contrast strength α) · `TOPK` (per-position top-K exported to the student) ·
`MODE=reverse` · `STEPS` · `NGPU` · `CONTEXT_DELTA={answer_only|solution_answer|trace}` (setting 3).

The proxy-teacher additions on top of verl:
- `verl/verl/workers/actor/proxy_teacher.py` — composes `z_base + α(z₊ − z₋)`, exports the top-K teacher.
- `verl/verl/workers/actor/dp_actor.py` — the student's per-position reverse-KL objective.
- `verl/verl/trainer/ppo/ref_input_utils.py` — builds the ± hint contexts (setting 3).

## 📊 Evaluation

```bash
# Math (avg@32): AIME24/25, HMMT-Feb/Nov
python evaluation/math_eval/eval_math.py --input_file data/aime24/test.jsonl \
    --model_path <merged_hf_ckpt> --output_file out.jsonl --n 32 --temperature 1.0 --max_tokens 30720

# Code: HumanEval+/MBPP+ (evaluation/evalplus) and LiveCodeBench (evaluation/LiveCodeBench)
```

Merge an FSDP checkpoint to Hugging Face format first:
```bash
python verl/scripts/legacy_model_merger.py merge --backend fsdp \
    --local_dir <ckpt>/actor --target_dir <hf_dir> --hf_model_path <base_model>
```

## 📖 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{yu2026w2sopd,
  title     = {Weak-to-Strong On-Policy Distillation},
  author    = {Yu, Fangxu and Lin, Zinan and Liu, Xiaodong and Xu, Weijia and Xu, Michael and Zhou, Tianyi and Gao, Jianfeng},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
```
