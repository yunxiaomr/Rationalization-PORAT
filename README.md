# 🧩 Learnable Game-theoretic Policy Optimization for Data-centric Self-explanation Rationalization

**Accepted by IEEE Transactions on Knowledge and Data Engineering (TKDE) [paper link](https://arxiv.org/pdf/2510.13393)**

This repository contains code for the paper "PORAT: Learnable Game-Theoretic Policy Optimization for Data-centric Self-Explanation Rationalization". 
We release some key code in experiments. We will release all the code used in experiments.
🚀 *Code is coming soon!*

## 📘 Overview
In this paper, we systematically revisit cooperative self-explanation rationalization from a novel game-theoretic perspective and identify the fundamental cause of rationale collapse. We then propose a novel approach, Game-theoretic Policy Optimization oriented RATionalization (PoRAT), which progressively introduces policy interventions to address the game equilibrium in the cooperative game process, thereby guiding the model toward a more optimal solution state.

## Environments
Ubuntu 22.04.4 LTS; NVIDIA RTX6000 Ada; CUDA 12.1; python 3.9.

We suggest you to create a virtual environment with: conda create -n PORAT python=3.9.0

Then activate the environment with: conda activate PORAT 

Install packages: pip install -r requirements.txt


## Datasets
Following previous research, we obtain BeerAdvocate, BeerAdvocate* and HotelReview datasets.
- BeerAdvocate. 
- BeerAdvocate*. 
- HotelReview. 

## Running example
### Beer-Aroma
Aroma: python -u main_porat.py --dis_lr 1 --hidden_dim 200 --data_type beer --freezing 2 --save 1 --dropout 0.2 --lr 0.0002 --batch_size 128 --gpu 1 --sparsity_percentage 0.175 --sparsity_lambda 11 --continuity_lambda 12 --epochs 100 --aspect 1 --correlated 1  --pretrain_agent True --reward_mode causal_effect --num_beam 4 --topK_ratio 0.1 --action_K 3 --pre_ward True  --writer  './results_final/beer_correlated/PORAT/noname1_20'  > ./results_final/beer_correlated/PORAT/noname1_20.log	

**_Notes_**: "--sparsity_percentage 0.175" means "$s=0.175$" in Sec.3 (But the actual sparsity is different from $s$. When you change the random seed, you need to adjust the "sparsity_percentage" according to the actual sparsity on the test set.). "--sparsity_lambda 11 --continuity_lambda 12 " means $\lambda_1=11, \lambda_2=12$. "--pretrain_agent True --reward_mode causal_effect --num_beam 4 --action_K 8 --pre_ward True  --topK_ratio 0.1 " means the parameters of related agent during a search process. 
"--epochs 100" means we run 100 epochs and take the results when the "dev_acc" is best.

## Result  
You will get the result like "best_dev_epoch=78" at last. Then you need to find the result corresponding to the epoch with number "78".  
For Beer-Palate, you may get a result like: 

Train time for epoch #78 : 
traning epoch:78 recall:0.8235 precision:0.8493 f1-score:0.8362 accuracy:0.8387
Validate
dev epoch:78 recall:0.7924 precision:0.7894 f1-score:0.7909 accuracy:0.7905
Validate Sentence
dev dataset : recall:0.8908 precision:0.7108 f1-score:0.7906 accuracy:0.7641
Annotation
annotation dataset : recall:0.8939 precision:0.9961 f1-score:0.9422 accuracy:0.8940

The annotation performance: sparsity: 19.1542, precision: 69.3768, recall: 85.2943, f1: 76.5165
Episode: 79, loss: 514.6271, cls loss: 309.7217, spa loss: 47.9385, con loss: 166.8680, rl loss: -9.9016, avg_reward: -0.0002

The line "The annotation performance: sparsity: 19.1542, precision: 69.3768, recall: 85.2943, f1: 76.5165" indicates that the rationale F1 score is 76.5165.


## Dependencies
- torch==2.1.0
- matplotlib==3.9.2
- numpy==1.26.3
- pandas==2.2.2
- scikit_learn==1.5.1
- seaborn==0.13.2
- tensorboardX==2.6.2.2
- protobuf==5.28.0



We provide:

* ✅ Fine-tuning **settings** (`.yaml` configuration file)
* ✅ Well-constructed **dataset and metadata** (`dataset_info.json`)

You can directly use these files in Llama-Factory to reproduce the  **Fact Decomposition LLM** .

## 🧠 Citation

If you find our work useful, please cite:

```
@ARTICLE{11271751,
  author={Zhao, Yunxiao and Wang, Zhiqiang and Yu, Xingtong and Li, Xiaoli and Liang, Jiye and Li, Ru},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={Learnable Game-Theoretic Policy Optimization for Data-Centric Self-Explanation Rationalization}, 
  year={2025},
  volume={},
  number={},
  pages={1-14},
  keywords={Generators;Predictive models;Games;Optimization;Correlation;Computational modeling;Benchmark testing;Standards;Sentiment analysis;Gold;Data-centric Explainability;Self-explanation;Rationale Mining;Game-theoretic Policy Optimization},
  doi={10.1109/TKDE.2025.3638864}}

```
