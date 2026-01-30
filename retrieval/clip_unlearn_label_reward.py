# coding=utf-8
import os
import sys
import json
import logging
import copy

import torch
import torch.nn.functional as F


import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.utils import now

from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners.runner_base import RunnerBase
from torch.utils.data import DataLoader
from torch import nn
from lavis.tasks import *

from params import parse_args
from clip_unlearn_reward import get_reward_model
from lavis_evaluate import setup_seeds
from torch import nn


from clip_unlearn_baseline import _encode_image_patches, _encode_text, _mask_to_patch_attention
from utils import (
    _tokenize_texts,
    _select_neg_texts_by_minsim,
    _get_logits_and_feats,
    _resolve_clip_model,
    _save_unlearned_checkpoint,
    prepare_dr_data,
    prepare_df_data,
    prepare_df_data_for_test,
    prepare_dr_data_for_test,
    _load_sam3_masks,
    _load_forget_train_ids,
    _load_forget_test_ids,
    eval_split_no_tta,
    _select_neg_texts_by_proximal,
    _dump_detailed_topk_results
)

# Allow running this file as a script by adding its directory to sys.path.
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)


def supervised_unlearn_train(
    cfg,
    model,
    teacher,
    df_train_loader,
    dr_train_loader,
    optimizer, scaler,
    lambda_rtf=1.0,          # Df 的“多模态解耦”损失（unlearn vs teacher@shuffled）
    lambda_keep=2.0,        # Dr 的“多模态保持”损失（unlearn vs teacher@matched）
    lambda_uni=0.1,         # 单模态保持（img/text embedding vs teacher）
    lambda_rdr=1.0,      # Dr 的 reward advantage 损失
    max_epoch=1,
    log_interval=50,
    neg_mode="shuffle",     # 新增：'shuffle' 或 'minsim'
    concept_token="horse",
    sam3_mask_dir=None,
    sam3_mask_suffix=".png",
    reward_model=None,
):
    device = model.device
    mse = nn.MSELoss()

    if not sam3_mask_dir:
        raise ValueError(
            "sam3_mask_dir must be provided to compute attention-guided loss.")
    concept_tokens = _tokenize_texts([concept_token]).to(device)
    clip_model = _resolve_clip_model(model)
    visual = clip_model.visual if hasattr(clip_model, "visual") else None
    if visual is None or not hasattr(visual, "image_size"):
        raise AttributeError(
            "CLIP visual encoder does not expose image_size for SAM3 mask mapping.")
    image_size = visual.image_size[0] if isinstance(
        visual.image_size, (tuple, list)) else visual.image_size

    # 以较短的一方为 epoch 基础步数
    iters_per_epoch = min(len(df_train_loader), len(dr_train_loader))
    for ep in range(max_epoch):
        model.train()
        df_iter = iter(df_train_loader)
        dr_iter = iter(dr_train_loader)
        running = {"rtf": 0.0, "rdr": 0.0, "uni": 0.0, "tot": 0.0}
        for it in range(iters_per_epoch):
            # ===== 1) 取 batch =====
            df_s = next(df_iter)
            dr_s = next(dr_iter)
            img_df = df_s["image"].to(device, non_blocking=True)
            img_dr = dr_s["image"].to(device, non_blocking=True)
            # ✅ 直接使用 dataloader 提供的 token ids（LongTensor）：
            # 兼容 list[str] 或 LongTensor；若是 list[str] 则现场 tokenize 成 [B, L] LongTensor
            txt_df = df_s.get("text_input", df_s.get("text", None))
            txt_dr = dr_s.get("text_input", dr_s.get("text", None))
            if not isinstance(txt_df, torch.Tensor):
                txt_df = _tokenize_texts(txt_df)          # -> LongTensor[B, L]
            if not isinstance(txt_dr, torch.Tensor):
                txt_dr = _tokenize_texts(txt_dr)          # -> LongTensor[B, L]
            txt_df = txt_df.to(device, non_blocking=True)
            txt_dr = txt_dr.to(device, non_blocking=True)
            df_image_paths = df_s.get("image_path")

            B = img_df.size(0)

            if neg_mode == "shuffle":
                # ✅ baseline：随机打乱
                perm = torch.randperm(B).cpu()
                df_text_neg = [txt_df[i] for i in perm.tolist()]
            elif neg_mode == "simrange":
                # ✅ ours：对每张图像选 teacher 相似度在某范围内的文本
                df_text_neg = _select_neg_texts_by_similarity_range(
                    teacher, img_df, txt_df, lower_percent=0, upper_percent=20)
            else:
                # ✅ ours：对每张图像选 teacher 最不相似的文本
                df_text_neg = _select_neg_texts_by_minsim(
                    teacher, img_df, txt_df)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=True):
                if df_image_paths is None:
                    raise KeyError(
                        "Df batch missing image_path; update dataset to return image_path for SAM3 masks.")

                sam3_masks = _load_sam3_masks(
                    df_image_paths, sam3_mask_dir, sam3_mask_suffix, image_size)  # (B, H, W)
                sam3_masks = sam3_masks.to(device, non_blocking=True)

                # grid 是每边 patch 数（比如 ViT-L/14 输入 336 时通常 grid=24，N=576
                patch_feats, grid = _encode_image_patches(model, img_df) # (B,N,D)，N 是补丁的数量
                text_feat = _encode_text(model, concept_tokens).squeeze(0) # (D,)
                # (B,N) = (B,N,D) · (D,)
                patch_sim = F.cosine_similarity(patch_feats, text_feat.unsqueeze(0), dim=-1) # 遗忘概念文本特征和整图特征
                patch_mask = _mask_to_patch_attention(sam3_masks, grid) # (B,N) 0/1
                patch_num = patch_mask.sum(dim=1) # (B,)，mask 区域有多少个 patch，保证不会因为 mask 大小影响 loss 尺度导致训练不稳
                masked_similarity = patch_sim * patch_mask # (B,N)
                # # 使用指数函数来放大相似度差异，惩罚相似度较高的区域
                # exp_similarity = torch.exp(masked_similarity)  # 使用指数函数，放大相似度差异
                # loss_rtf = (exp_similarity.sum(dim=1) / patch_num.clamp(min=1.0)).mean()
                loss_rtf = (masked_similarity.sum(dim=1) / patch_num).mean()

                loss_rdr = torch.tensor(0.0, device=device)
                if reward_model is not None:
                    with torch.no_grad():
                        # 先按 patch 粒度抽取背景区域（复用上面的 grid/patch_mask）
                        bg_patch_mask = 1.0 - patch_mask  # (B, N)
                        bg_patch_mask = bg_patch_mask.view(B, grid, grid)  # (B, g, g)
                        bg_mask = F.interpolate(
                            bg_patch_mask.unsqueeze(1),
                            size=img_df.shape[-2:],
                            mode="nearest",
                        )  # (B, 1, H, W)
                        syn_img = img_df * bg_mask
                        reward_img = syn_img
                        reward_target = reward_model.clip_model.visual.image_size
                        if isinstance(reward_target, (tuple, list)):
                            reward_target = reward_target[0]
                        if reward_img.shape[-1] != reward_target:
                            reward_img = F.interpolate(
                                reward_img,
                                size=reward_target,
                                mode="bicubic",
                                align_corners=True,
                            )
                        reward_model.set_image_features(images=reward_img)
                        if isinstance(txt_dr, torch.Tensor):
                            reward_model.set_text_features(
                                tokenized_cap=txt_dr)
                        else:
                            reward_model.set_text_features(captions=txt_dr)
                        text_pool = reward_model.text_features
                        image_pool = reward_model.image_features
                        if text_pool is None or image_pool is None:
                            raise RuntimeError(
                                "Reward model text/image features are not initialized.")
                        text_index_pool = torch.arange(
                            text_pool.size(0), device=text_pool.device)
                        image_index_pool = torch.arange(
                            image_pool.size(0), device=image_pool.device)
                        clip_scores = reward_model.CLIPScore(
                            text_index=text_index_pool,
                            images_index=image_index_pool,
                            pairwise=True,
                        )
                        clip_probs = torch.softmax(clip_scores, dim=-1)
                        sample_k = min(reward_model.sample_k,
                                       clip_scores.size(1))
                        topk_probs, indices = torch.topk(
                            clip_probs, sample_k, dim=-1, largest=True)
                        adv = topk_probs.detach()

                    sim_i2t_syn_u, sim_t2i_syn_u, img_syn_u, txt_syn_u = _get_logits_and_feats(
                        model, syn_img, txt_dr
                    )
                    rep_output = torch.repeat_interleave(
                        sim_i2t_syn_u, sample_k, dim=0)  # [B*sample_k, N_text]
                    text_index = indices.flatten()  # [B*sample_k]
                    ce = F.cross_entropy(rep_output, text_index, reduction="none").view(
                        img_df.size(0), sample_k)
                    loss_rdr = (adv * ce).mean()

                sim_i2t_dr_u, sim_t2i_dr_u, img_dr_u, txt_dr_u = _get_logits_and_feats(
                    model, img_dr, txt_dr
                )
                with torch.no_grad():
                    sim_i2t_dr_t, sim_t2i_dr_t, img_dr_t, txt_dr_t = _get_logits_and_feats(
                        teacher, img_dr, txt_dr
                    )

                # ===== 3) Dr：保持一致（与原模型 matched 分布接近）=====
                with torch.no_grad():
                    sim_i2t_dr_t, sim_t2i_dr_t, img_dr_t, txt_dr_t = _get_logits_and_feats(
                        teacher, img_dr, txt_dr
                    )
                loss_keep = mse(sim_i2t_dr_u, sim_i2t_dr_t) + \
                    mse(sim_t2i_dr_u, sim_t2i_dr_t)
                # loss_keep = mse(sim_i2t_dr_u, sim_i2t_dr_t) + mse(sim_t2i_dr_u, sim_t2i_dr_t)

                # ===== 4) 单模态保持（避免 encoder 漂移过大）=====
                loss_uni = mse(img_dr_u, img_dr_t) + mse(txt_dr_u, txt_dr_t)

                loss = lambda_rtf * loss_rtf + lambda_keep * loss_keep + \
                    lambda_rdr * loss_rdr + lambda_uni * loss_uni

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # log
            running["rtf"] += float(loss_rtf.detach().item())
            running["rdr"] += float(loss_rdr.detach().item())
            running["uni"] += float(loss_uni.detach().item())
            running["tot"] += float(loss.detach().item())
            if (it + 1) % log_interval == 0:
                t = it + 1
                logging.info(
                    f"[EP {ep+1}/{max_epoch}] it={t}/{iters_per_epoch} "
                    f"rtf={running['rtf']/t:.4f} rdr={running['rdr']/t:.4f} "
                    f"uni={running['uni']/t:.4f} total={running['tot']/t:.4f}"
                )
        logging.info(f"[EP {ep+1}] done.")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"
    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()
    args = parse_args()
    print('\n job_ID {}: \n'.format(job_id))

    cfg = Config(args)
    cfg.forget_train_file = args.forget_train_file
    cfg.forget_test_file = args.forget_test_file

    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)
    # set after init_distributed_mode() to only log on master.
    setup_logger()

    cfg.pretty_print()
    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)

    # Prepare for Dr and Df
    data_name = list(cfg.datasets_cfg.keys())[0]
    if 'flickr30k' in data_name:
        data_type = 'flickr30k'
    elif 'coco' in data_name:
        data_type = 'coco'
    elif data_name == 'nlvr':
        data_type = 'nlvr'
    elif 'snli_ve' in data_name:
        data_type = 've'

    dtrain = datasets[data_name]['train']
    dtest = datasets[data_name]['test']

    forget_train_ids, forget_train_id_set, _ = _load_forget_train_ids(
        dtrain, cfg, data_type)
    forget_test_ids, forget_test_id_set, _ = _load_forget_test_ids(
        dtest, cfg, data_type)
    print(f"Loaded {len(forget_test_ids)} forget test images.")
    dr = prepare_dr_data(dtrain, cfg, data_type,
                         df_ids_set=forget_train_id_set)
    df = prepare_df_data(dtrain, cfg, data_type,
                         df_ids=forget_train_ids, df_ids_set=forget_train_id_set)
    df_for_test = prepare_df_data_for_test(
        dtrain, dtest, cfg, data_type, df_ids=forget_test_ids, df_ids_set=forget_test_id_set
    )

    retain_sample_size = len(df_for_test.annotation*5)  # 建议 *5，和后续 AUC 切片一致

    dr_for_test = prepare_dr_data_for_test(
        dtrain,
        dtest,
        cfg,
        data_type,
        retain_sample_size,
        df_ids_set=forget_test_id_set,
    )

    datasets[data_name]['df'] = df_for_test
    datasets[data_name]['dr'] = dr_for_test

    runner = RunnerBase(cfg=cfg, job_id=job_id, task=task,
                        model=model, datasets=datasets)
    df_loader = runner.dataloaders['df']
    dr_loader = runner.dataloaders['dr']
    device = runner.model.device

    # —— 将日志同时写入文件 —— #
    os.makedirs(args.output, exist_ok=True)
    log_path = os.path.join(args.output, f"run_{now()}.log")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"))
    logging.getLogger().addHandler(file_handler)
    logging.info(f"[Log] Saving logs to: {log_path}")

    # ========= 新增：判断是否只评测原始模型 =========
    use_original_only = bool(getattr(args, "original_eval", False)) or (
        getattr(args, "unlearn_method", "") in [
            "original", "none", "clip-original"]
    )

    if not use_original_only:
        # 只有在需要做遗忘训练时才初始化 scaler / optimizer / teacher / train_loader
        # setup automatic mixed-precision (Amp) loss scaling
        scaler = torch.cuda.amp.GradScaler(init_scale=1000)

        # create policy model（遗忘模型 / 被训练的模型）
        print("unlearn policy arch:", args.arch)

        # # https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.AdamW
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, eps=1e-06, weight_decay=args.weight_decay)
        optim_state = copy.deepcopy(optimizer.state_dict())

        # 冻结“原模型”作为 teacher
        teacher = copy.deepcopy(model).to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False

        # 训练用 dataloader（直接走 LAVIS dataset 的 __getitem__）
        bs_train = cfg.run_cfg.batch_size_train
        num_workers = cfg.run_cfg.num_workers

        logging.info(f"[Train] Df train samples: {len(df)}")
        logging.info(f"[Train] Dr train samples: {len(dr)}")
        logging.info(f"[Train] Batch size: {bs_train}")

        df_train_loader = DataLoader(
            df, batch_size=bs_train, shuffle=True, num_workers=num_workers, drop_last=True)
        dr_train_loader = DataLoader(
            dr, batch_size=bs_train, shuffle=True, num_workers=num_workers, drop_last=True)

        logging.info("[Train] Using Multidelete-style supervised baseline.")
        logging.info(
            "[Train] Using MSE-based supervised baseline (shuffle/minsim).")
        reward_model = get_reward_model(device, args)
        # ========= 监督式 unlearning（替代原 TTA 训练），评测仍用 df/dr/test =========
        supervised_unlearn_train(
            cfg, model, teacher, df_train_loader, dr_train_loader,
            optimizer, scaler,
            lambda_rtf=getattr(args, "lambda_df", 1.0),          # Df 多模态解耦损失权重
            lambda_keep=getattr(args, "lambda_dr", 2.0),        # Dr 多模态保持损失权重
            lambda_uni=getattr(args, "lambda_uni", 0.1),        # 单模态 MSE 权重
            # Dr reward advantage 损失权重
            lambda_rdr=getattr(args, "lambda_rdr", 1.0),
            max_epoch=getattr(args, "max_epoch", 1),
            log_interval=getattr(args, "log_interval", 50),
            neg_mode=args.neg_mode,
            concept_token=getattr(args, "concept_token", "horse"),
            sam3_mask_dir=getattr(args, "sam3_mask_dir", None),
            sam3_mask_suffix=getattr(args, "sam3_mask_suffix", ".png"),
            reward_model=reward_model,
        )
    else:
        logging.info(
            "[Eval-only] Skip unlearning. Directly evaluate ORIGINAL CLIP on df/dr/test.")

    # 监督训练完后，再在 df/dr/test 上按原规则评测（不做 TTA）
    task_name = "i2t" if args.retrieval_task == "image2text" else "t2i"
    # 确保 df_loader 只加载遗忘集的图像
    print(
        f"df_loader contains {len(df_loader.dataset.image)} images from the forget test set.")
    score_df = eval_split_no_tta(df_loader, model, task=task_name, text_bs=128)
    score_dr = eval_split_no_tta(dr_loader, model, task=task_name, text_bs=128)

    # 调用评估函数
    eval_df = task._report_metrics(
        score_df, score_df.T, df_loader.dataset.txt2img, df_loader.dataset.img2txt)
    eval_dr = task._report_metrics(
        score_dr, score_dr.T, dr_loader.dataset.txt2img, dr_loader.dataset.img2txt)

    # 输出
    for name, result in [("df", eval_df), ("dr", eval_dr)]:
        output_filename = os.path.join(
            args.output, f"results_{args.retrieval_task}_{name}.json")
        logging.info(output_filename)
        result = {k: round(v, 3) for k, v in result.items()}
        logging.info(result)
        with open(output_filename, "w") as fp:
            json.dump(result, fp, indent=4)

    # 训练后再根据评测得到的分数矩阵，导出 Top-K 结果
    task_suffix = "i2t" if args.retrieval_task == "image2text" else "t2i"
    df_res_path = os.path.join(
        args.output, f"detailed_top10_{task_suffix}_df.jsonl")
    _dump_detailed_topk_results(
        score_df, df_loader, task_name, df_res_path, k=10)
    dr_res_path = os.path.join(
        args.output, f"detailed_top10_{task_suffix}_dr.jsonl")
    _dump_detailed_topk_results(
        score_dr, dr_loader, task_name, dr_res_path, k=10)

    if not use_original_only and args.save_unlearned_model and get_rank() == 0:
        save_dir = os.path.join(args.output, args.unlearned_subdir)
        _save_unlearned_checkpoint(
            model,
            save_dir,
            args.unlearned_model_name,
            args.unlearned_meta_name,
            args.cfg_path,
            args_dict=vars(args),
            job_id=job_id,
        )

    # 用训完的模型评测官方 Flickr30k 测试集（全量）
    # 用 runner 自带的 'test' dataloader
    test_loader = runner.dataloaders.get('test', None)
    if test_loader is not None:
        task_name = "i2t" if args.retrieval_task == "image2text" else "t2i"
        full_test_scores = eval_split_no_tta(
            test_loader, model, task=task_name, text_bs=128)
        # 汇报标准 Recall@K
        eval_test = task._report_metrics(
            full_test_scores,
            full_test_scores.T,
            test_loader.dataset.txt2img,
            test_loader.dataset.img2txt
        )
        test_out = os.path.join(
            args.output, f"results_{args.retrieval_task}_official_test.json")
        logging.info(test_out)
        with open(test_out, "w") as fp:
            json.dump({k: round(v, 3)
                      for k, v in eval_test.items()}, fp, indent=4)
        logging.info(
            f"[OFFICIAL TEST] { {k: round(v, 3) for k, v in eval_test.items()} }")
    else:
        logging.warning(
            "[OFFICIAL TEST] runner.dataloaders['test'] 不存在，已跳过官方测试集评测。")


if __name__ == "__main__":
    main()
