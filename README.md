# MindSimulator

## Introduction

**Concept-selective regions** within the human cerebral cortex exhibit significant activation in response to specific visual stimuli associated with particular concepts (Left of the following figure). Precisely localizing these regions stands as a crucial long-term goal in neuroscience to grasp essential brain functions and mechanisms. Conventional experiment-driven approaches hinge on manually constructed visual stimulus collections and corresponding brain activity recordings, constraining the support and coverage of concept localization. Additionally, these stimuli (Right of the following figure) often consist of concept objects in unnatural contexts and are potentially biased by subjective preferences, thus prompting concerns about the validity and generalizability of the identified regions. 

![Image has missed!](./Figs/Introduction.png)

To address these limitations, we propose a data-driven exploration approach. By synthesizing extensive brain activity recordings, we statistically localize various concept-selective regions. Our proposed **_MindSimulator_** (as following images) leverages advanced generative technologies to learn the probability distribution of brain activity conditioned on concept-oriented visual stimuli. 

![Image has missed!](./Figs/model.png)

This enables the creation of simulated brain recordings that reflect real neural response patterns. Using the synthetic recordings, we successfully localize several well-studied concept-selective regions and validate them against empirical findings, achieving promising prediction accuracy. We further localized the arbitrary concept-selective regions (as following images). 

![Image has missed!](./Figs/Localization.png)

The feasibility opens avenues for exploring novel concept-selective regions and provides prior hypotheses for future neuroscience research.

For more details, please refer to our conference paper:

arXiv: [https://arxiv.org/abs/2503.02351](https://arxiv.org/abs/2503.02351)

OpenReview: [https://openreview.net/forum?id=vgt2rSf6al](https://openreview.net/forum?id=vgt2rSf6al)


## Results of synthetic fMRI



## MindSimulator's Train & Inference

**Step 1: Preparation**

Our code is based on MindEye and MindEye2. Therefore, you should first follow the [**MindEye2 repository**](https://github.com/MedARC-AI/MindEyeV2) to set up the virtual environment and generative model. Then, you should integrate our modified codes with the [**MindEye2 weight files**](https://huggingface.co/datasets/pscotti/mindeyev2/tree/main) and organize them according to the following directory structure:

- Encoding/
  - mindeye2_src/
    - generative_models/...
    - evals/...
    - train_logs/...
    - wds/...
    - Other MindEye2's Files
    - Our Modified Codes in "mindeye2_src"
  - Our Other Codes

**Step 2: Training**

You should run _voxel_autoencoder_aligning.py_ to train the fMRI autocoder and then run _voxel_diffusion_prior.py_ to train the diffusion estimator.


**Step 3: Inference**


## Concept-selective Region Localization



## Acknowledgements

Our research is based on [MindEye](https://papers.nips.cc/paper_files/paper/2023/hash/4ddab70bf41ffe5d423840644d3357f4-Abstract-Conference.html), [MindEye2](https://openreview.net/forum?id=65XKBGH5PO), and [BrainDIVE](https://papers.nips.cc/paper_files/paper/2023/hash/ef0c0a23a1a8219c4fc381614664df3e-Abstract-Conference.html). We sincerely appreciate their significant contributions to the research community.

## Citations

If our work helps your research, please consider citing our paper, thanks!

```
@inproceedings{bao2025mindsimulator,
  title={MindSimulator: Exploring Brain Concept Localization via Synthetic fMRI},
  author={Bao, Guangyin and Zhang, Qi and Gong, Zixuan and Wu, Zhuojia and Miao, Duoqian},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```

