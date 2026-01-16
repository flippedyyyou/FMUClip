"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import logging
import os

import numpy as np
import torch
from lavis.common.dist_utils import is_main_process
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask


@registry.register_task("retrieval")
class RetrievalTask(BaseTask):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        return cls(cfg=run_cfg)

    def evaluation(self, model, data_loader, **kwargs):
        # score_i2t, score_t2i = model.compute_sim_matrix(model, data_loader)
        score_i2t, score_t2i = model.compute_sim_matrix(data_loader, task_cfg=self.cfg)

        if is_main_process():
            eval_result = self._report_metrics(
                score_i2t,
                score_t2i,
                data_loader.dataset.txt2img,
                data_loader.dataset.img2txt,
                # add
                texts=getattr(data_loader.dataset, "text", None),
                concept_token=getattr(self.cfg, "concept_token", None),
            )
            logging.info(eval_result)
        else:
            eval_result = None

        return eval_result

    def after_evaluation(self, val_result, **kwargs):
        return val_result

    @staticmethod
    @torch.no_grad()
    def _report_metrics(scores_i2t, scores_t2i, txt2img, img2txt):

        # Images->Text
        ranks = np.zeros(scores_i2t.shape[0])
        for index, score in enumerate(scores_i2t):
            inds = np.argsort(score)[::-1]
            # Score
            rank = 1e20
            for i in img2txt[index]:
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank

        # Compute metrics
        tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

        # Text->Images
        ranks = np.zeros(scores_t2i.shape[0])

        for index, score in enumerate(scores_t2i):
            inds = np.argsort(score)[::-1]
            ranks[index] = np.where(inds == txt2img[index])[0][0]

        # Compute metrics
        ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
        ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
        ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

        tr_mean = (tr1 + tr5 + tr10) / 3
        ir_mean = (ir1 + ir5 + ir10) / 3
        r_mean = (tr_mean + ir_mean) / 2

        agg_metrics = (tr1 + tr5 + tr10) / 3

        eval_result = {
            "txt_r1": tr1,
            "txt_r5": tr5,
            "txt_r10": tr10,
            "txt_r_mean": tr_mean,
            "img_r1": ir1,
            "img_r5": ir5,
            "img_r10": ir10,
            "img_r_mean": ir_mean,
            "r_mean": r_mean,
            "agg_metrics": agg_metrics,
        }
        with open(
            os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
        ) as f:
            f.write(json.dumps(eval_result) + "\n")
        return eval_result

    # def _report_metrics(scores_i2t, scores_t2i, txt2img, img2txt, keyword=None):
    #     """
    #     计算并返回检索指标，包括关键词召回率和不包含关键词的 ground truth 召回率。
    #     """
    #     # Images->Text
    #     ranks = np.zeros(scores_i2t.shape[0])
    #     for index, score in enumerate(scores_i2t):
    #         inds = np.argsort(score)[::-1]
    #         # Score
    #         rank = 1e20
    #         for i in img2txt[index]:
    #             tmp = np.where(inds == i)[0][0]
    #             if tmp < rank:
    #                 rank = tmp
    #         ranks[index] = rank

    #     # Compute metrics
    #     tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    #     tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    #     tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    #     # Text->Images
    #     ranks = np.zeros(scores_t2i.shape[0])

    #     for index, score in enumerate(scores_t2i):
    #         inds = np.argsort(score)[::-1]
    #         ranks[index] = np.where(inds == txt2img[index])[0][0]

    #     # Compute metrics
    #     ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    #     ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    #     ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    #     tr_mean = (tr1 + tr5 + tr10) / 3
    #     ir_mean = (ir1 + ir5 + ir10) / 3
    #     r_mean = (tr_mean + ir_mean) / 2

    #     agg_metrics = (tr1 + tr5 + tr10) / 3

    #     keyword_ranks_1 = 0
    #     keyword_ranks_5 = 0
    #     keyword_ranks_10 = 0
    #     non_keyword_gt_ranks_1 = 0
    #     non_keyword_gt_ranks_5 = 0
    #     non_keyword_gt_ranks_10 = 0

    #     for index, score in enumerate(scores_i2t):
    #         inds = np.argsort(score)[::-1]  # 每张图片检索的文本相似度从大到小降序排序

    #         # 统计关键词召回率和非关键词ground truth召回率
    #         keyword_found_1 = keyword_found_5 = keyword_found_10 = 0
    #         non_keyword_found_1 = non_keyword_found_5 = non_keyword_found_10 = 0
            
    #         for i in inds[:10]:  # 检查排名前10的文本
    #             caption = txt2img[i]  # 获取文本内容
    #             print(caption)
    #             is_ground_truth = i in img2txt[index]  # 判断该文本是否为 ground truth

    #             if isinstance(caption, str):  # 确保caption是字符串
    #                 # 检查关键词是否在文本中
    #                 if keyword and keyword.lower() in caption.lower():
    #                     print("Found keyword in caption:", caption)
    #                     if i in inds[:1]: keyword_found_1 += 1
    #                     if i in inds[:5]: keyword_found_5 += 1
    #                     if i in inds[:10]: keyword_found_10 += 1

    #                 # 检查是否为不包含关键词的ground truth
    #                 if keyword and keyword.lower() not in caption.lower() and is_ground_truth:
    #                     if i in inds[:1]: non_keyword_found_1 += 1
    #                     if i in inds[:5]: non_keyword_found_5 += 1
    #                     if i in inds[:10]: non_keyword_found_10 += 1

    #         # 计算关键词召回率
    #         keyword_ranks_1 += keyword_found_1 / 1
    #         keyword_ranks_5 += keyword_found_5 / 5
    #         keyword_ranks_10 += keyword_found_10 / 10
            
    #         # 计算非关键词ground truth召回率
    #         non_keyword_gt_ranks_1 += non_keyword_found_1 / 1
    #         non_keyword_gt_ranks_5 += non_keyword_found_5 / 5
    #         non_keyword_gt_ranks_10 += non_keyword_found_10 / 10

    #     # 计算每个指标的平均值
    #     keyword_r1 = 100.0 * keyword_ranks_1 / len(scores_i2t)
    #     keyword_r5 = 100.0 * keyword_ranks_5 / len(scores_i2t)
    #     keyword_r10 = 100.0 * keyword_ranks_10 / len(scores_i2t)
        
    #     non_keyword_gt_r1 = 100.0 * non_keyword_gt_ranks_1 / len(scores_i2t)
    #     non_keyword_gt_r5 = 100.0 * non_keyword_gt_ranks_5 / len(scores_i2t)
    #     non_keyword_gt_r10 = 100.0 * non_keyword_gt_ranks_10 / len(scores_i2t)

    #     # 最终评估结果
    #     eval_result = {
    #         "txt_r1": tr1,
    #         "txt_r5": tr5,
    #         "txt_r10": tr10,
    #         "txt_r_mean": tr_mean,
    #         "img_r1": ir1,
    #         "img_r5": ir5,
    #         "img_r10": ir10,
    #         "img_r_mean": ir_mean,
    #         "r_mean": r_mean,
    #         "agg_metrics": agg_metrics,
    #         "keyword_r1": keyword_r1,
    #         "keyword_r5": keyword_r5,
    #         "keyword_r10": keyword_r10,
    #         "non_keyword_gt_r1": non_keyword_gt_r1,
    #         "non_keyword_gt_r5": non_keyword_gt_r5,
    #         "non_keyword_gt_r10": non_keyword_gt_r10,
    #     }

    #     # 保存评估结果
    #     with open(
    #         os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a"
    #     ) as f:
    #         f.write(json.dumps(eval_result) + "\n")

    #     return eval_result
