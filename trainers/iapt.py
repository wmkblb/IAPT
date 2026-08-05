import json
import os.path as osp
import random
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from clip.model import QuickGELU
from clip.model import convert_weights
from torch.onnx.symbolic_opset9 import new_ones

from .imagenet_templates import IMAGENET_TEMPLATES
from collections import OrderedDict
import math
import time

_tokenizer = _Tokenizer()

CoPrompt_dataset_name_mapping = {
    "Caltech101": "caltech",
    "DescribableTextures": "dtd",
    "EuroSAT": "eurosat",
    "FGVCAircraft": "fgvc",
    "Food101": "food101",
    "ImageNet": "imagenet",
    "ImageNetA": "imagenet_a",
    "ImageNetR": "imagenet_r",
    "ImageNetSketch": "imagenet_sketch",
    "ImageNetV2": "imagenetv2",
    "OxfordFlowers": "oxford_flowers",
    "OxfordPets": "oxford_pets",
    "StanfordCars": "stanford_cars",
    "SUN397": "sun397",
    "UCF101": "ucf101",
}


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def load_clip_to_cpu_teacher(cfg, zero_shot_model=False):
    backbone_name = cfg.TRAINER.IAPT.TEACHER_NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    print(f"CLIP Teacher name is {backbone_name}")

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    # Return original CLIP model for generating frozen VL features
    design_details = {"trainer": 'IVLP',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}
    model = clip.build_model(state_dict or model.state_dict(), design_details)
    return model

def load_clip_to_cpu(cfg, zero_shot_model=False):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    if not zero_shot_model:
        design_details = {"trainer": 'IAPT',
                          "vision_depth": cfg.TRAINER.IAPT.PROMPT_DEPTH,
                          "language_depth": cfg.TRAINER.IAPT.PROMPT_DEPTH,
                          "vision_ctx": cfg.TRAINER.IAPT.N_CTX,
                          "language_ctx": cfg.TRAINER.IAPT.N_CTX}
        model = clip.build_model(state_dict or model.state_dict(), design_details)
    else:
        # Return original CLIP model for generating frozen VL features
        design_details = {"trainer": 'IVLP',
                          "vision_depth": 0,
                          "language_depth": 0, "vision_ctx": 0,
                          "language_ctx": 0}
        model = clip.build_model(state_dict or model.state_dict(), design_details)
        return model
    return model


# ================================================================================================================

attribute_features = torch.load("Lexicon/attribute.pt", map_location="cuda")  # (27888,512)

# concept_features = torch.load("Lexicon/concept.pt", map_location="cuda")  # (8,512)

cluster_ids = torch.load("Lexicon/cluster_k3000/cluster_ids.pt", map_location="cuda").long()


TOP_L = 10
TOP_K = 5

PRE_TOPN = 30
ALPHA = 0.20
BETA = 0.05

TOP_POS = 4
# ===================================!!!!!!!!!!!!!!!!!!!!!=======================
def sample_gumbel(shape, device, dtype=torch.float32, eps=1e-20):
    U = torch.rand(shape, device=device, dtype=dtype)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, tau=1.0, hard=False, dim=-1):
    orig_dtype = logits.dtype
    logits_fp32 = logits.float()

    g = sample_gumbel(logits_fp32.shape, logits_fp32.device, dtype=logits_fp32.dtype)
    y = F.softmax((logits_fp32 + g) / tau, dim=dim)

    if hard:
        index = y.max(dim=dim, keepdim=True)[1]
        y_hard = torch.zeros_like(y).scatter_(dim, index, 1.0)
        y = y_hard - y.detach() + y

    return y.to(orig_dtype)


def deterministic_topk_weights(logits, k):
    B, L = logits.shape
    k = min(k, L)

    idx = torch.topk(logits, k=k, dim=1).indices
    weights = torch.zeros_like(logits)
    weights.scatter_(1, idx, 1.0)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-12)
    return weights


def gumbel_topk_weights(logits, k, tau=1.0, hard=True, training=True):
    B, L = logits.shape
    k = min(k, L)

    if not training:
        return deterministic_topk_weights(logits, k)

    orig_dtype = logits.dtype
    work_logits = logits.float()
    neg_inf = torch.finfo(work_logits.dtype).min

    selected_mask = torch.zeros_like(work_logits, dtype=torch.bool)
    all_weights = []

    for _ in range(k):
        masked_logits = work_logits.masked_fill(selected_mask, neg_inf)

        w = gumbel_softmax_sample(masked_logits, tau=tau, hard=hard, dim=1)
        all_weights.append(w)

        selected_idx = w.argmax(dim=1, keepdim=True)

        selected_onehot = torch.zeros_like(selected_mask)
        selected_onehot = selected_onehot.scatter(1, selected_idx, True)
        selected_mask = selected_mask | selected_onehot

    weights = torch.stack(all_weights, dim=1).sum(dim=1)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-12)

    return weights.to(orig_dtype)
# ===================================!!!!!!!!!!!!!!!!!!!!!=======================


def diverse_select_with_cluster_batch(
    global_scores_all,      # [B, M]
    attribute_features,
    cluster_ids,            # [M]
    top_l=20,
    pre_topn=50,
    alpha=0.20,
    beta=0.05,
):
    assert global_scores_all.dim() == 2, f"global_scores_all 应为 [B, M]，但得到 {global_scores_all.shape}"
    assert attribute_features.dim() == 2, f"attribute_features 应为 [M, D]，但得到 {attribute_features.shape}"
    assert cluster_ids.dim() == 1, f"cluster_ids 应为 [M]，但得到 {cluster_ids.shape}"

    device = global_scores_all.device
    B, M = global_scores_all.shape

    assert attribute_features.shape[0] == M, \
        f"attribute_features 数量与 global_scores_all 不一致: {attribute_features.shape[0]} vs {M}"
    assert cluster_ids.shape[0] == M, \
        f"cluster_ids 数量与 global_scores_all 不一致: {cluster_ids.shape[0]} vs {M}"


    attribute_features = attribute_features.to(device=device)
    cluster_ids = cluster_ids.to(device=device).long()
    global_scores_all = global_scores_all.to(device=device)

    pre_topn = min(pre_topn, M)
    top_l = min(top_l, pre_topn)


    # pre_scores:  [B, pre_topn]
    # pre_indexes: [B, pre_topn]
    pre_scores, pre_indexes = torch.topk(global_scores_all, k=pre_topn, dim=1)

    all_selected_scores = []
    all_selected_indexes = []
    all_selected_cluster_ids = []


    for b in range(B):
        scores_b = pre_scores[b]                # [pre_topn]
        indexes_b = pre_indexes[b]              # [pre_topn]
        feats_b = attribute_features[indexes_b] # [pre_topn, D]
        cids_b = cluster_ids[indexes_b]         # [pre_topn]


        sim_matrix = feats_b @ feats_b.T        # [pre_topn, pre_topn]


        selected_mask = torch.zeros(pre_topn, device=device, dtype=torch.bool)


        max_redundancy = torch.zeros(pre_topn, device=device, dtype=scores_b.dtype)


        selected_local = []


        first = 0
        selected_local.append(first)
        selected_mask[first] = True


        max_redundancy = torch.maximum(max_redundancy, sim_matrix[:, first])


        for _ in range(1, top_l):
            selected_cids = cids_b[selected_mask]   # [num_selected]


            # [pre_topn, num_selected] -> [pre_topn]
            cluster_repeat_mask = (cids_b.unsqueeze(1) == selected_cids.unsqueeze(0)).any(dim=1)
            cluster_repeat_mask = cluster_repeat_mask.to(dtype=scores_b.dtype)


            current_scores = scores_b - alpha * max_redundancy - beta * cluster_repeat_mask


            current_scores = current_scores.masked_fill(selected_mask, torch.finfo(current_scores.dtype).min)

            best_idx = torch.argmax(current_scores).item()

            selected_local.append(best_idx)
            selected_mask[best_idx] = True


            max_redundancy = torch.maximum(max_redundancy, sim_matrix[:, best_idx])

        selected_local = torch.tensor(selected_local, device=device, dtype=torch.long)

        selected_scores_b = scores_b[selected_local]    # [top_l]
        selected_indexes_b = indexes_b[selected_local]  # [top_l]
        selected_cids_b = cids_b[selected_local]        # [top_l]

        all_selected_scores.append(selected_scores_b)
        all_selected_indexes.append(selected_indexes_b)
        all_selected_cluster_ids.append(selected_cids_b)


    selected_scores = torch.stack(all_selected_scores, dim=0)           # [B, top_l]
    selected_indexes = torch.stack(all_selected_indexes, dim=0)         # [B, top_l]
    selected_cluster_ids = torch.stack(all_selected_cluster_ids, dim=0) # [B, top_l]

    return selected_scores, selected_indexes, selected_cluster_ids

# ===============================================================================================================
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, cross_prompts_text_deeper):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
          # 4,n_cls,512     L,n_cls,dim

        combined = [x, cross_prompts_text_deeper]
        outputs = self.transformer(combined)
        x = outputs[0]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class CrossModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.IAPT.N_CTX
        ctx_init = cfg.TRAINER.IAPT.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        # Default is 1, which is compound shallow prompting
        assert cfg.TRAINER.IAPT.PROMPT_DEPTH >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg.TRAINER.IAPT.PROMPT_DEPTH  # max=12, but will create 11 such shared prompts
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print('MaPLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")
        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        self.ctx = nn.Parameter(ctx_vectors)

        ctx_vectors_vision = torch.empty(n_ctx, 768, dtype=dtype)
        nn.init.normal_(ctx_vectors_vision, std=0.02)
        self.ctx_vision = nn.Parameter(ctx_vectors_vision)
        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # Minimum can be 1, which defaults to shallow MaPLe
        # compound prompts
        self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
                                                      for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_text:
            nn.init.normal_(single_para, std=0.02)
        # Also make corresponding projection layers, for each prompt
        self.compound_prompts_vision = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 768))
                                                       for _ in range(self.compound_prompts_depth - 1)])
        for single_para in self.compound_prompts_vision:
            nn.init.normal_(single_para, std=0.02)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

        # visual
        clip_model_temp = load_clip_to_cpu(cfg, True).float().cuda()
        clip_model_temp_image = load_clip_to_cpu_teacher(cfg, True)
        with torch.no_grad():
            self.ZS_image_encoder = clip_model_temp_image.visual
        # text
        with open(f"gpt_file/{CoPrompt_dataset_name_mapping[cfg.DATASET.NAME]}_prompt.json") as f:
            gpt3_prompt = json.load(f)
        print("\nGetting textual features as CLIP's classifier.")
        clip_weights = gpt_clip_classifier(
            classnames, gpt3_prompt, clip_model_temp, cfg.DATASET.NAME
        )
        self.fixed_embeddings = clip_weights

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)


        return prompts, self.ctx_vision, self.compound_prompts_text, self.compound_prompts_vision

class VisibleAttributeMemory(nn.Module):
    def __init__(self, feature_dim=512, memory_size=32, dtype=torch.float16, momentum=0.9):
        super().__init__()
        self.feature_dim = feature_dim
        self.memory_size = memory_size
        self.momentum = momentum
        self.dtype = dtype

        cache = torch.randn(memory_size, feature_dim)
        cache = F.normalize(cache, dim=-1)


        self.register_buffer("cache", cache.to(dtype=dtype))

    def read(self, text_features):
        """
        text_features: [n_cls, D]

        return:
            memory_summary: [n_cls, D]
            read_weights:   [n_cls, R]
        """
        text_norm = F.normalize(text_features.float(), dim=-1)
        cache = F.normalize(self.cache.float(), dim=-1)


        scores = text_norm @ cache.t()              # [n_cls, R]
        read_weights = F.softmax(scores, dim=-1)    # [n_cls, R]

        memory_summary = read_weights @ cache       # [n_cls, D]
        memory_summary = F.normalize(memory_summary, dim=-1)

        return memory_summary.to(dtype=text_features.dtype), read_weights.detach()

    @torch.no_grad()
    def write(self, attr_tokens, token_weights=None):
        if attr_tokens is None:
            return

        if attr_tokens.dim() == 3:
            B, L, D = attr_tokens.shape
            tokens = attr_tokens.reshape(-1, D)

            if token_weights is not None:
                weights = token_weights.reshape(-1).float()
            else:
                weights = torch.ones(tokens.shape[0], device=tokens.device, dtype=torch.float32)

        elif attr_tokens.dim() == 2:
            tokens = attr_tokens
            D = tokens.shape[-1]

            if token_weights is not None:
                weights = token_weights.reshape(-1).float()
            else:
                weights = torch.ones(tokens.shape[0], device=tokens.device, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported attr_tokens shape: {attr_tokens.shape}")


        valid_mask = weights > 1e-6
        if valid_mask.sum() == 0:
            return

        tokens = tokens[valid_mask]
        weights = weights[valid_mask]

        tokens = F.normalize(tokens.float(), dim=-1)
        cache = F.normalize(self.cache.float(), dim=-1)


        scores = tokens @ cache.t()      # [N, R]
        assign = scores.argmax(dim=-1)   # [N]

        updated_cache = cache.clone()

        for r in range(self.memory_size):
            idx = (assign == r).nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue

            selected_tokens = tokens[idx]       # [Nr, D]
            selected_weights = weights[idx]     # [Nr]


            sim = (selected_tokens @ cache[r].unsqueeze(-1)).squeeze(-1)  # [Nr]


            w = F.softmax(sim, dim=0) * selected_weights
            w = w / w.sum().clamp(min=1e-12)
            w = w.unsqueeze(-1)

            proto = (w * selected_tokens).sum(dim=0)
            proto = F.normalize(proto, dim=-1)


            updated_cache[r] = self.momentum * cache[r] + (1.0 - self.momentum) * proto
            updated_cache[r] = F.normalize(updated_cache[r], dim=-1)

        self.cache.copy_(updated_cache.to(device=self.cache.device, dtype=self.cache.dtype))

    def forward(self, text_features, attr_tokens=None, token_weights=None):

        if self.training and attr_tokens is not None:
            if token_weights is not None:
                self.write(attr_tokens.detach(), token_weights.detach())
            else:
                self.write(attr_tokens.detach(), None)

        memory_summary, read_weights = self.read(text_features)
        return memory_summary, read_weights


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = CrossModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.lambd = cfg.TRAINER.IAPT.LAMBD
        # ==========================================================================================
        self.text_refiner = nn.Linear(clip_model.visual.output_dim * 2,
                                      clip_model.visual.output_dim)  # 2*512 ---> 512
        self.text_refiner = self.text_refiner.to(dtype=self.dtype)
        self.text_alpha = nn.Parameter(torch.tensor(-2.0))


        self.visual_refiner = nn.Linear(clip_model.visual.output_dim * 2,
                                      clip_model.visual.output_dim)
        self.visual_refiner = self.visual_refiner.to(dtype=self.dtype)
        self.visual_alpha = nn.Parameter(torch.tensor(-2.0))

        nn.init.zeros_(self.text_refiner.weight)
        nn.init.zeros_(self.text_refiner.bias)
        nn.init.zeros_(self.visual_refiner.weight)
        nn.init.zeros_(self.visual_refiner.bias)

        #self.concept_router = nn.Linear(clip_model.visual.output_dim, 8)
        #self.concept_router = self.concept_router.to(dtype=self.dtype)

        #nn.init.normal_(self.concept_router.weight, std=0.02)
        #nn.init.zeros_(self.concept_router.bias)

        # =========================================================
        # Visible Attribute Memory


        # =========================================================
        self.visible_attr_memory = VisibleAttributeMemory(
            feature_dim=clip_model.visual.output_dim,
            memory_size=32,
            dtype=self.dtype,
            momentum=0.9,
        )

        # semantic_anchor = ratio * concept_summary + (1-ratio) * memory_summary

        # self.memory_anchor_ratio = 0.7

        #!!!!!!!!!!!!!!!!
        self.gumbel_tau_fine = 1.0
        self.gumbel_hard_fine = True
        # !!!!!!!!!!!!!!!!!!

        # =========================================================

        # attribute_features: [27888, 512]
        # concept_features:   [8, 512]
        # cluster_ids:        [27888]
        # =========================================================
        attr = attribute_features
        if not isinstance(attr, torch.Tensor):
            attr = torch.tensor(attr)
        attr = attr / attr.norm(dim=-1, keepdim=True).clamp(min=1e-12)

        # con = concept_features
        # if not isinstance(con, torch.Tensor):
        #     con = torch.tensor(con)
        # con = con / con.norm(dim=-1, keepdim=True).clamp(min=1e-12)

        cid = cluster_ids
        if not isinstance(cid, torch.Tensor):
            cid = torch.tensor(cid)
        cid = cid.long()
        # ======================================================================================
        self.register_buffer("attr_features", attr)
        # self.register_buffer("con_features", con)
        self.register_buffer("cluster_ids_buf", cid)

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()


        attr_features = self.attr_features.to(dtype=self.dtype)
        # con_features = self.con_features.to(dtype=self.dtype)
        cluster_ids = self.cluster_ids_buf

        # =========================================================

        # =========================================================
        with torch.no_grad():
            global_image_features_fixed, patch_image_features_fixed = self.prompt_learner.ZS_image_encoder(
                image.type(self.dtype), return_all=True
            )
            global_image_features_fixed = global_image_features_fixed / global_image_features_fixed.norm(
                dim=-1, keepdim=True
            ).clamp(min=1e-12)
            patch_image_features_fixed = patch_image_features_fixed / patch_image_features_fixed.norm(
                dim=-1, keepdim=True
            ).clamp(min=1e-12)

            # [B, 512] @ [512, 27888] -> [B, 27888]
            global_scores_all = global_image_features_fixed @ attr_features.T


            coarse_scores, coarse_indexes, coarse_cluster_ids = diverse_select_with_cluster_batch(
                global_scores_all=global_scores_all,   # [B, 27888]
                attribute_features=attr_features,      # [27888, 512]
                cluster_ids=cluster_ids,               # [27888]
                top_l=TOP_L,
                pre_topn=PRE_TOPN,
                alpha=ALPHA,
                beta=BETA,
            )

            # =====================================================

            # coarse_indexes: [B, L]
            # attr_features:  [M, D]
            # -> candidate_attr_feat: [B, L, D]
            # =====================================================
            candidate_attr_feat = attr_features[coarse_indexes]  # [B, L, 512]

            # patch_image_features_fixed: [B, P, D]
            # candidate_attr_feat.transpose(1,2): [B, D, L]
            # -> patch_scores: [B, P, L]
            patch_scores = torch.matmul(
                patch_image_features_fixed, candidate_attr_feat.transpose(1, 2)
            )


            top_pos_scores = patch_scores.topk(k=TOP_POS, dim=1).values.mean(dim=1)  # [B, L]


            mean_scores = patch_scores.mean(dim=1)  # [B, L]

            # local score
            local_scores = top_pos_scores - mean_scores  # [B, L]

            # # final score
            # final_scores = coarse_scores.to(local_scores.dtype) * local_scores  # [B, L]
            #
            #
            #
            #
            #

            # final_top_scores, final_top_pos = final_scores.topk(k=TOP_K, dim=1)  # [B, K]
            #

            # final_attr_indexes = torch.gather(coarse_indexes, dim=1, index=final_top_pos)  # [B, K]
            #

            # final_attr_feats = attr_features[final_attr_indexes]  # [B, K, 512]
            #
            # attr_weight = torch.softmax(final_top_scores, dim=1).unsqueeze(-1)  # [B, K, 1]
            # attr_summary = (attr_weight * final_attr_feats).sum(dim=1)  # [B, 512]
            # final_attr_feats = attr_summary / attr_summary.norm(dim=-1, keepdim=True).clamp(min=1e-12)  # B 512

            # final score
            final_scores = coarse_scores.to(local_scores.dtype) * local_scores  # [B, L]

            # =====================================================


            # =====================================================
            fine_weights = gumbel_topk_weights(
                logits=final_scores,
                k=TOP_K,
                tau=self.gumbel_tau_fine,
                hard=self.gumbel_hard_fine,
                training=self.training,
            )  # [B, L]


            attr_weight = fine_weights.unsqueeze(-1)  # [B, L, 1]
            attr_summary = (attr_weight * candidate_attr_feat).sum(dim=1)  # [B, 512]
            final_attr_feats = attr_summary / attr_summary.norm(dim=-1, keepdim=True).clamp(min=1e-12)

            visible_attr_tokens = candidate_attr_feat
            visible_attr_token_weights = fine_weights


        # =========================================================
        # 3. Prompted image/text features
        # =========================================================
        text_input, visual_ctx, cross_prompts_text_deeper, cross_prompts_visual_deeper = self.prompt_learner()

        text_features = self.text_encoder(
            text_input, tokenized_prompts, cross_prompts_text_deeper)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)

        # ------------------------------------------------------------------------------------------
        # concept_logits = self.concept_router(text_features)  # [n_cls, 8]
        # concept_weights = torch.softmax(concept_logits, dim=-1)  # [n_cls, 8]
        #
        # concept_summary = concept_weights @ con_features  # [n_cls, 512]
        # concept_summary = concept_summary / concept_summary.norm(dim=-1, keepdim=True).clamp(min=1e-12)

        # -----------------------------------------------------------------


        image_features = self.image_encoder(
            image.type(self.dtype), visual_ctx, cross_prompts_visual_deeper)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)


        fusion_input = torch.cat([image_features, final_attr_feats.to(image_features.dtype)], dim=-1)  # [B,512*2]
        refined_delta = self.visual_refiner(fusion_input)  # B 512
        alpha = torch.sigmoid(self.visual_alpha).to(image_features.dtype)
        image_features = alpha * refined_delta +  image_features  # B 512
        image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)


        # concept_summary = concept_summary.expand(text_features.size(0), -1)  # [n_cls, 512]

        # =========================================================
        # Visible Attribute Memory read


        # =========================================================
        # =========================================================
        # Visible Attribute Memory read


        # =========================================================
        memory_summary, memory_read_weights = self.visible_attr_memory(
            text_features,
            attr_tokens=visible_attr_tokens if self.training else None,
            token_weights=visible_attr_token_weights if self.training else None,
        )

        # =========================================================


        # =========================================================
        semantic_anchor = memory_summary.to(text_features.dtype)
        semantic_anchor = semantic_anchor / semantic_anchor.norm(dim=-1, keepdim=True).clamp(min=1e-12)

        # =========================================================
        # Visible-Attribute-Memory guided text refinement
        # =========================================================
        text_fusion_input = torch.cat([text_features, semantic_anchor], dim=-1)  # [n_cls, 1024]
        text_delta = self.text_refiner(text_fusion_input)  # [n_cls, 512]
        alpha_t = torch.sigmoid(self.text_alpha).to(text_features.dtype)

        text_features = alpha_t * text_delta + text_features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)


        # =========================================================

        # =========================================================
        # image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        image_features = image_features + global_image_features_fixed
        image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)

        # text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        text_features = text_features + self.prompt_learner.fixed_embeddings.to(dtype=text_features.dtype)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)

        # logits
        logits = logit_scale * image_features @ text_features.t()

        if self.prompt_learner.training:
            loss_cls = F.cross_entropy(logits, label)

            text_features_fixed = self.prompt_learner.fixed_embeddings.to(dtype=text_features.dtype)
            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-7)

            score_text = cos(text_features, text_features_fixed)
            loss_distill_text = 1.0 - torch.mean(score_text)

            score_image = cos(image_features, global_image_features_fixed)
            loss_distill_image = 1.0 - torch.mean(score_image)

            loss_distill = loss_distill_text + loss_distill_image
            return loss_cls + self.lambd * loss_distill

        return logits

def gpt_clip_classifier(classnames, gpt_prompts, clip_model, dataset_name):
    import os
    os.makedirs("cache/", exist_ok=True)

    with torch.no_grad():
        clip_weights = []
        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace("_", " ")
            texts = []
            for t in gpt_prompts[classname]:
                texts.append(t)
            texts = clip.tokenize(texts)
            if torch.cuda.is_available():
                clip_model = clip_model.cuda()
                texts = texts.cuda()
            # prompt ensemble
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings /= class_embeddings.norm()
            clip_weights.append(class_embeddings)

        clip_weights = torch.stack(clip_weights, dim=0)
        if torch.cuda.is_available():
            clip_weights = clip_weights.cuda()
        torch.save(clip_weights, f"cache/{dataset_name}_clip_weights_random.pt")
    return clip_weights

@TRAINER_REGISTRY.register()
class IAPT(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.IAPT.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.IAPT.PREC == "fp32" or cfg.TRAINER.IAPT.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")

        trainable_names = (
            "prompt_learner",
            "VPT",
            "text_refiner",
            "visual_refiner",
        )

        for name, param in self.model.named_parameters():
            requires_grad = any(key in name for key in trainable_names)


            if "ZS_image_encoder" in name:
                requires_grad = False

            param.requires_grad_(requires_grad)


        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        print("=" * 60)
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable parameters: {trainable_params / 1e3:.3f}K")
        print(f"Trainable parameters: {trainable_params / 1e6:.6f}M")
        print("=" * 60)

        print("\nTrainable parameter details:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(f"{name:80s} {param.numel():,}")


        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("VLPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.IAPT.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.IAPT.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

