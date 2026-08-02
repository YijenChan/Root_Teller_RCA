<div align="center">

# ThinkFL: Self-Refining Failure Localization for Microservice Systems via Reinforcement Fine-Tuning

[![Paper](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2504.18776)

</div>

> [!NOTE]  
> This repository is built upon **OpenRLHF** and includes several modifications. The primary changes are concentrated in `openrlhf/trainer/ppo_utils/experience_maker.py`, along with a few additional minor adjustments. Overall, the execution pipeline follows the standard workflow of OpenRLHF.

The main entry scripts are:
- `thinkfl/train_sft_aiops.sh`, which is used for **SFT-based fine-tuning**.
- `thinkfl/train_grpo_aiops.sh`, which is used for **GRPO-based fine-tuning**.

## Citation
```
@article{zhang2025thinkfl,
  title={ThinkFL: Self-Refining Failure Localization for Microservice Systems via Reinforcement Fine-Tuning},
  author={Zhang, Lingzhe and Zhai, Yunpeng and Jia, Tong and Duan, Chiming and Yu, Siyu and Gao, Jinyang and Ding, Bolin and Wu, Zhonghai and Li, Ying},
  journal={arXiv preprint arXiv:2504.18776},
  year={2025}
}
```