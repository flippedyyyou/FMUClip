# README


## Motivation
### Disadvantages of Current Methods
+ 现有方法是image-level
    + 当图像中存在多个concept（复杂图像）时，无法对target concept进行细粒度的精确遗忘。
    + Image-level会导致模型遗忘图像背景中的non-target concepts。

## Task Definition
给定一个target concept $c_t$以及其对应的图像$i$，目标是在遗忘$c_t$的同时，不遗忘非目标概念$C_{nt}$，特别是遗忘训练集中的背景concept $C_{bg} (C_{bg} \subset C_{nt})$。

## Early Study
+ 将待遗忘图像对应的文本label替换为最不相似的文本：$D_f$低，同时$D_r$很高。

## Baseline
+ 仅仅遗忘了遗忘集中的图片所对应的文本标签，但并没有真正遗忘目标概念。

## TODO
### Dataset Construction
#### Candidate datasets
+ flickr30k_entities
+ MS COCO

#### Splitting Strategy
+ **flickr30k_entities:** 基于CIFAR-100中已经定义的class，去flickr中选取包含相应concept的image，筛选图片数>=200的concept作为forgetting set以及retain set的training set，从CIFAR-100中选择concept相对应的image作为test set。

+ **Forgetting set:** Select complex images containing *multiple* concepts (items) as training set, and images with less concepts as test set.
+ **Retain set:** TBD.

#### Next Steps
+ Analyze the distribution of each target concept across raw datasets, as well as the number of other non-target concepts present in the corresponding images.

### Loss Design
#### Region-level Forgetting
+ 使用local / patch level的视觉向量与目标概念计算相似度是否存在粒度不匹配的问题

#### Reward Module
+ 当前K的取值为1，且top K的选取范围为一个training batch，是否存在噪声/漏选问题

#### Umimodal Constraint
+ Loss的数值的数量级与另外2个loss不匹配

### Tasks
#### classification
+ unlearn_classification训练集测试集划分受Mask影响混乱：现按10%从cifar100按class抽样出遗忘集训练候选集（50/class），mask提取置信度超过阈值的会过滤一部分得到jsonl里的遗忘集，其中70%作为遗忘集训练集，30%*10%作为测试集；保留集为非遗忘class的全部，其中70%作为保留集训练集（但batch里取了遗忘集训练集和保留集训练集最小值），30%*10%作为测试集。
+ baseline：已改完，还未测试效果最好的minsim方法指标
