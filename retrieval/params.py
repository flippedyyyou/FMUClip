# coding=utf-8
import os
import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Test Time Adaptation for Retrieval Task")
    parser.add_argument("--precision", choices=["amp", "fp16", "fp32"], default="amp",
                            help="Floating point precition.")

    parser.add_argument('--output', type=str, default='tta_ret_rl_01', help='the output path')
    parser.add_argument('--retrieval_task', type=str, default="image2text", choices=["image2text", "text2image"],
                            help='using simple average or exponential average for gradient update')
    parser.add_argument('--arch', type=str, default='ViT-B-16', help='model architecture')

    # RL config
    parser.add_argument('--tta_steps', default=1, type=int, help='test-time-adapt steps')
    parser.add_argument('--lr', '--learning-rate', default=1e-6, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--weight_decay', default=5e-4, type=float)

    parser.add_argument('--sample_k', type=int, default=5)
    parser.add_argument('--multiple_reward_models', type=int, default=0)
    parser.add_argument('--reward_arch', type=str, default='ViT-L-14')
    parser.add_argument('--reward_process', type=int, default=1,
                         help='If true, process rewards (raw CLIPScore)')
    parser.add_argument('--process_batch', type=int, default=0,
                         help='If true, process rewards through the whole batch (augmentations from a single images)')
    parser.add_argument('--reward_amplify', type=int, default=0)
    parser.add_argument('--weighted_scores', type=int, default=1)

    # args of momentum_update
    parser.add_argument('--momentum_update', type=int, default=0,
                         help='If true, update the model in a momentum fashion')
    parser.add_argument('--update_freq', type=int, default=256)
    parser.add_argument('--update_w', type=float, default=1.0)
    parser.add_argument('--tta_momentum', type=float, default=0.9999)
    
    # LAVIS confif
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    parser.add_argument('--lambda_df', type=float, default=1.0, help='weight for forget set reward')
    parser.add_argument('--lambda_dr', type=float, default=2.0, help='weight for retain set reward')
    parser.add_argument('--drift_coef', type=float, default=0.01, help='retain feature drift regularization coefficient')
    parser.add_argument('--lambda_uni', type=float, default=0.01, help='unimodal regularization coefficient')
    parser.add_argument("--max_epoch", type=int, default=1, help="number of train epochs")
    parser.add_argument("--neg_mode", type=str, default="shuffle", choices=["shuffle", "minsim", "simrange"],
                        help="How to build negatives for Df: shuffle (baseline) or minsim (teacher argmin) or simrange.")
    parser.add_argument("--cliperase", action="store_true", help="use ClipErase-style supervised baseline") #当命令行里出现这个参数时，把它设为 True；如果不写，则保持默认值 False。除非你显式写 default=True，否则默认就是 False
    parser.add_argument("--original_eval", action="store_true", help="use Clip original eval setting")
    parser.add_argument('--unlearn_method', type=str, default='', help='strategy flag to pick unlearning pipeline')
    parser.add_argument('--concept_token', type=str, default='dog', help='concept keyword used for attention masking')
    parser.add_argument('--attn_threshold', type=float, default=0.2, help='threshold for pruning high-attention patches')
    parser.add_argument('--reward_topk', type=int, default=5, help='top-k captions retrieved by the BLIP reward model')
    parser.add_argument('--lambda_attn', type=float, default=1.0, help='coefficient for attention-guided loss')
    parser.add_argument('--lambda_fea', type=float, default=1.0, help='coefficient for feature regularization loss')
    parser.add_argument('--lambda_sim', type=float, default=1.0, help='coefficient for similarity KL loss')
    parser.add_argument('--lambda_reward', type=float, default=1.0, help='coefficient for reward advantage loss')
    parser.add_argument('--sam3_mask_dir', type=str, default='', help='directory containing SAM3 binary masks')
    parser.add_argument('--sam3_mask_suffix', type=str, default='.png', help='file suffix for SAM3 masks')
    # checkpoint export
    parser.add_argument(
        "--save_unlearned_model",
        action="store_true",
        help="Save the fine-tuned/unlearned CLIP checkpoint after running ClipErase",
    )
    parser.add_argument(
        "--unlearned_model_name",
        type=str,
        default="cliperase_unlearned.pt",
        help="Filename of the saved unlearned CLIP checkpoint",
    )
    parser.add_argument(
        "--unlearned_meta_name",
        type=str,
        default="cliperase_unlearned_meta.json",
        help="Filename of the metadata json that records how the checkpoint was produced",
    )
    parser.add_argument(
        "--unlearned_subdir",
        type=str,
        default="unlearned_clip",
        help="Subdirectory under --output to place the saved ClipErase checkpoint and metadata",
    )

    parser.add_argument('--forget_train_file', type=str, default='', help='path to the forget train ids file')
    parser.add_argument('--forget_test_file', type=str, default='', help='path to the forget test ids file')

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    print('\n', vars(args), '\n')
    save_hp_to_json(args.output, args)

    return args


def save_hp_to_json(directory, args):
    """Save hyperparameters to a json file
    """
    filename = os.path.join(directory, 'hparams_{}.json'.format(args.retrieval_task))
    hparams = vars(args)
    with open(filename, 'w') as f:
        json.dump(hparams, f, indent=4, sort_keys=True)
