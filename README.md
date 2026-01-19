# README

## Early Study
+ 将待遗忘图像对应的文本label替换为最不相似的文本：$D_f$低，同时$D_r$很高。

## Hypothesis
+ 


## Baseline
+ 仅仅遗忘了遗忘集中的图片所对应的文本标签，但并没有真正遗忘目标概念。

## TODO
### Region-level Forgetting
+ 使用local / patch level的视觉向量与目标概念计算相似度是否存在粒度不匹配的问题

### Reward Module
+ 当前K的取值为1，且top K的选取范围为一个training batch，是否存在噪声/漏选问题

### Umimodal Constraint
+ Loss的数值的数量级与另外2个loss不匹配