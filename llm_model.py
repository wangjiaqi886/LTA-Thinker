import json
import os
from typing import List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.cache_utils import Cache
from peft import PeftModel, LoraConfig
from fastNLP import logger
from transformers import TrainerCallback, Trainer


class MappingLayerTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if self.args.process_index == 0:
            os.makedirs(output_dir, exist_ok=True)
            model_to_save = (
                self.model.module if hasattr(self.model, "module") else self.model
            )
            torch.save(
                model_to_save.projection.state_dict(),
                os.path.join(output_dir, "projection.bin"),
            )
            for name, module in model_to_save.named_modules():
                if isinstance(module, TransformerBlockWithRMSNorm):
                    torch.save(
                        model_to_save.TransformerBlock.state_dict(),
                        os.path.join(output_dir, "TransformerBlock.bin"),
                    )


class SavePretrainedCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def is_world_process_zero(self):
        # 检查当前进程是否是主进程
        return (
            torch.distributed.get_rank() == 0
            if torch.distributed.is_initialized()
            else True
        )

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if not self.is_world_process_zero():
            return
        # 构造保存路径，按 epoch 编号区分
        epoch = int(state.epoch)
        save_directory = os.path.join(self.output_dir, f"epoch_{int(epoch)}")

        # 调用 model.save_pretrained 保存模型
        print(f"Saving model at epoch {epoch} to {save_directory}")
        model.save_pretrained(save_directory)
        save_detail = []
        os.makedirs(save_directory, exist_ok=True)

        # torch.save(model.projection.state_dict(), os.path.join(save_directory, 'projection.bin'))
        # save_detail.append('Projection Module')
        # logger.info(f'Saving parameters of projection module, includes: {[k for k, v in model.projection.state_dict().items()]}')
        #
        # logger.info(f'Successfully saved [{", ".join(save_detail)}] to dir `{save_directory}`.')


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习的缩放参数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()  # 防止数值不稳定问题，转换为 float32
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.rsqrt(mean_square + self.eps)
        x = x * rms
        return (self.weight * x).to(dtype=dtype)  # 保持原始数据类型


class TransformerBlockWithRMSNorm(nn.Module):
    def __init__(
        self, d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1, eps=1e-6
    ):
        super(TransformerBlockWithRMSNorm, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
            # dtype=torch.bfloat16,
        )
        self.linear1 = nn.Linear(
            d_model,
            dim_feedforward,
            # dtype=torch.bfloat16,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(
            dim_feedforward,
            d_model,
            # dtype=torch.bfloat16,
        )

        self.norm1 = RMSNorm(d_model, eps=eps)
        self.norm2 = RMSNorm(d_model, eps=eps)

        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src: torch.Tensor, src_mask=None, src_key_padding_mask=None):
        # Step 1: Self-Attention
        # Ensure input dtype matches self_attn weights (usually float32)
        src = src.to(self.self_attn.in_proj_weight.dtype)
        src2 = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout2(src2)
        src = self.norm1(src)

        # Step 2: Feed-Forward Network
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)

        return src


class EfficientSoftCoTFromSmallModel(nn.Module):

    def __init__(
        self,
        small_language_model_id,
        large_language_model_id,
        num_thought_tokens=None,
        tune_assistant_model=False,
        tune_base_model=False,
        path_to_projection_module=None,
        path_to_attention_module=None,
        path_to_small_language_model=None,
        path_to_large_language_model=None,
        sft_loss_weight=0.33,
        contrastive_loss_weight=0.33,
        KL_loss_weight=0.33,
        run_type="train",
        model_type=(False, "qwen"),  # 'qwen' or 'llama'
        **kwargs,
    ):
        super().__init__()
        common_kwargs = {
            "torch_dtype": torch.bfloat16,
            "_fast_init": False,
        }

        if run_type == "eval":
            common_kwargs["device_map"] = "auto"

        self.base_model = AutoModelForCausalLM.from_pretrained(
            large_language_model_id, **common_kwargs
        )
        self.config = AutoConfig.from_pretrained(
            large_language_model_id,
        )
        self.base_tokenizer = AutoTokenizer.from_pretrained(
            large_language_model_id,
        )

        self.model_type = model_type
        device = self.device
        if self.model_type[0]:
            self.assistant_model = AutoModelForCausalLM.from_pretrained(
                small_language_model_id, **common_kwargs
            )
            self.assistant_tokenizer = AutoTokenizer.from_pretrained(
                small_language_model_id,
            )
            if (
                path_to_small_language_model is not None
                and path_to_small_language_model not in ["None"]
            ):
                self.assistant_model.load_state_dict(
                    torch.load(path_to_small_language_model, weights_only=True)
                )
                logger.info(
                    f"Load weights from file `{path_to_small_language_model}` for assistant model."
                )
                self.assistant_model.to(device)
            for n, p in self.assistant_model.named_parameters():
                p.requires_grad = tune_assistant_model
            self.projection = nn.Linear(
                self.assistant_model.config.hidden_size,
                self.base_model.config.hidden_size,
                dtype=torch.bfloat16,
            )
            # self.projection = nn.Sequential(
            #     nn.Linear(
            #         self.assistant_model.config.hidden_size,
            #         self.base_model.config.hidden_size,
            #     ),
            #     nn.ReLU(),
            #     nn.Linear(
            #         self.base_model.config.hidden_size,
            #         self.base_model.config.hidden_size,
            #     ),
            #     nn.ReLU(),
            #     nn.Linear(
            #         self.base_model.config.hidden_size,
            #         self.base_model.config.hidden_size,
            #     ),
            #     nn.ReLU(),
            #     nn.Linear(
            #         self.base_model.config.hidden_size,
            #         self.base_model.config.hidden_size,
            #     ),
            # ).to(dtype=torch.bfloat16)
        else:
            self.assistant_model = None
            self.assistant_tokenizer = self.base_tokenizer

            if self.model_type[1] == "llama":
                self.eps = 1e-5
            elif self.model_type[1] == "qwen" or self.model_type[1] == "qwen3":
                self.eps = 1e-6
            self.TransformerBlock = TransformerBlockWithRMSNorm(
                d_model=self.config.hidden_size,
                nhead=1,
                dim_feedforward=self.config.intermediate_size,
                dropout=0.0,
                eps=self.eps,
            )
            self.projection = nn.Linear(
                self.base_model.config.hidden_size,
                self.base_model.config.hidden_size,
                dtype=torch.bfloat16,
            )
            # self.projection = nn.Sequential(
            #     nn.Linear(
            #         self.base_model.config.hidden_size,
            #         self.base_model.config.hidden_size,
            #     ),
            #     nn.ReLU(),
            #     nn.Linear(
            #         self.base_model.config.hidden_size,
            #         self.base_model.config.hidden_size,
            #     ),
            # ).to(dtype=torch.bfloat16)
            if (
                path_to_attention_module is not None
                and path_to_attention_module not in ["None"]
            ):
                self.TransformerBlock.load_state_dict(
                    torch.load(
                        path_to_attention_module, map_location="cpu", weights_only=True
                    )
                )
                logger.info(
                    f"Load weights from file `{path_to_attention_module}` for attention module."
                )
                self.TransformerBlock.to(device)

        for n, p in self.base_model.named_parameters():
            p.requires_grad = tune_base_model

        # self.base_tokenizer.add_special_tokens({'additional_special_tokens': ['<|extra_0|>','<|extra_1|>','<|extra_2|>','<|extra_3|>','<|extra_4|>','<|extra_5|>','<|extra_6|>','<|extra_7|>','<|extra_8|>','<|extra_9|>']})
        # self.assistant_tokenizer.add_special_tokens({'additional_special_tokens': ['<|extra_0|>','<|extra_1|>','<|extra_2|>','<|extra_3|>','<|extra_4|>','<|extra_5|>','<|extra_6|>','<|extra_7|>','<|extra_8|>','<|extra_9|>']})
        self.contrastive_loss_weight = contrastive_loss_weight
        self.KL_loss_weight = KL_loss_weight
        self.sft_loss_weight = sft_loss_weight
        self.num_thought_tokens = num_thought_tokens
        self.tune_assistant_model = tune_assistant_model
        self.tune_base_model = tune_base_model

        # LoRA configuration
        lora_config = LoraConfig(
            r=16,  # Rank
            lora_alpha=32,  # Scaling factor
            target_modules=[
                "q_proj",
                "v_proj",
            ],  # Modules to apply LoRA (depends on your model)
            lora_dropout=0.1,  # Dropout probability
            bias="none",  # Type of bias ("none", "all", or "lora_only")
            task_type="CAUSAL_LM",  # Task type (e.g., "SEQ2SEQ_LM", "CAUSAL_LM", etc.)
        )
        if tune_assistant_model:
            self.assistant_model = PeftModel(self.assistant_model, lora_config)
            logger.info(f"LoRA assistant model.")
        if tune_base_model:
            self.base_model = PeftModel(self.base_model, lora_config)
            logger.info(f"LoRA base model.")

        if path_to_projection_module is not None and path_to_projection_module not in [
            "None"
        ]:
            self.projection.load_state_dict(
                torch.load(
                    path_to_projection_module, map_location="cpu", weights_only=True
                )
            )
            logger.info(
                f"Load weights from file `{path_to_projection_module}` for projection module."
            )
            self.projection.to(device)

        if (
            path_to_large_language_model is not None
            and path_to_large_language_model not in ["None"]
        ):
            self.base_model.load_state_dict(
                torch.load(path_to_large_language_model, weights_only=True)
            )
            logger.info(
                f"Load weights from file `{path_to_large_language_model}` for base model."
            )
            self.base_model.to(device)

    @property
    def device(self):
        return self.base_model.device

    def save_pretrained(self, save_model_dir_root: str, **kwargs):
        save_detail = []
        os.makedirs(save_model_dir_root, exist_ok=True)
        if self.tune_base_model:
            # if hasattr(self.base_model, 'module'):  # DDP包装后模型属性
            #     self.base_model.module.save_pretrained(os.path.join(save_model_dir_root, "base_model"))
            # else:
            #     self.base_model.save_pretrained(os.path.join(save_model_dir_root, "base_model"))
            base_model_file = os.path.join(save_model_dir_root, "base_model.bin")
            logger.info(f"Saving base model to `{base_model_file}`")
            torch.save(self.base_model.state_dict(), base_model_file)
            save_detail.append("Base Model")

        if self.tune_assistant_model:
            # if hasattr(self.assistant_model, 'module'):
            #     self.assistant_model.module.save_pretrained(os.path.join(save_model_dir_root, "assistant_model"))
            # else:
            #     self.assistant_model.save_pretrained(os.path.join(save_model_dir_root, "assistant_model"))
            assistant_model_file = os.path.join(
                save_model_dir_root, "assistant_model.bin"
            )
            logger.info(f"Saving assistant model to `{assistant_model_file}`")
            torch.save(self.assistant_model.state_dict(), assistant_model_file)
            save_detail.append("Assistant Model")

        torch.save(
            self.projection.state_dict(),
            os.path.join(save_model_dir_root, "projection.bin"),
        )
        save_detail.append("Projection Module")
        logger.info(
            f"Saving parameters of projection module, includes: {[k for k, v in self.projection.state_dict().items()]}"
        )

        logger.info(
            f'Successfully saved [{", ".join(save_detail)}] to dir `{save_model_dir_root}`.'
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        thought_index: Optional[torch.LongTensor] = None,
        q_and_step_index_list: Optional[torch.LongTensor] = None,
        ans_ids: Optional[torch.LongTensor] = None,
        assistant_input_ids: Optional[torch.LongTensor] = None,
        assistant_attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        print_index=False,
    ):
        assert input_ids.device.type == "cuda", f"Input on {input_ids.device}!"
        # assert self.projection[0].weight.device.type == "cuda", f"Projection on {self.projection.weight.device}!"
        # assert (
        #     self.projection.weight.device.type == "cuda"
        # ), f"Projection on {self.projection.weight.device}!"
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        batch_size, seq_len = input_ids.size()

        if seq_len > 1:
            if input_ids is not None and inputs_embeds is None:
                inputs_embeds = self.base_model.get_input_embeddings()(input_ids)

            inputs_embeds, con_loss, kl_loss = self.get_inputs_embeds_for_base_model(
                assistant_input_ids,
                assistant_attention_mask,
                input_ids,
                inputs_embeds,
                thought_index,
                q_and_step_index_list,
                ans_ids,
                print_index,
            )

            outputs_from_llm = self.base_model(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )
        else:
            outputs_from_llm = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                inputs_embeds=inputs_embeds,
                labels=labels,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )
            con_loss = torch.tensor(0.0, device=outputs_from_llm.loss.device)
            kl_loss = torch.tensor(0.0, device=outputs_from_llm.loss.device)

        if labels is not None:
            sft_loss = outputs_from_llm.loss
            total_loss = (
                self.sft_loss_weight * sft_loss
                + self.contrastive_loss_weight * con_loss
                + self.KL_loss_weight * kl_loss
            )
        else:
            total_loss = None

        if not return_dict:
            output = (outputs_from_llm.logits, outputs_from_llm.past_key_values)
            if total_loss is not None:
                output = (total_loss,) + output
            return output
        else:
            return {
                "loss": total_loss,
                "logits": outputs_from_llm.logits,
                "past_key_values": outputs_from_llm.past_key_values,
                "hidden_states": (
                    outputs_from_llm.hidden_states if output_hidden_states else None
                ),
                "attentions": (
                    outputs_from_llm.attentions if output_attentions else None
                ),
            }

    def contrastive_loss(self, outputs, q_and_step_index_list, ans_ids):
        """
        Args:
            outputs: 模型输出，形状为 [batch_size, seq_len, hidden_dim]
            q_and_step_index_list: 二维张量，形状为 [batch_size, num_steps]

        Returns:
            loss: 对比学习损失值（标量）
        """
        batch_size, seq_len, hidden_dim = outputs.shape
        device = outputs.device
        assistant_question_end_index = q_and_step_index_list[:, 0]
        assistant_step_index = q_and_step_index_list[:, 1:]

        # 提取锚点 token
        anchor_indices = (
            assistant_question_end_index.unsqueeze(1)
            .unsqueeze(2)
            .expand(-1, 1, hidden_dim)
        )
        anchor = torch.gather(outputs, dim=1, index=anchor_indices).squeeze(
            1
        )  # [batch_size, hidden_dim]

        ans_embeddings = self.base_model.get_input_embeddings()(ans_ids)
        pad_token_id = self.base_tokenizer.pad_token_id
        ans_mask = (ans_ids != pad_token_id).unsqueeze(-1)  # [B, L, 1]
        ans_emb = (ans_embeddings * ans_mask).sum(dim=1)  # [B, H]
        ans_emb /= ans_mask.sum(dim=1).clamp(min=1e-9)  # 防止除以0

        # 提取候选 token
        num_steps = assistant_step_index.size(1)
        step_indices = assistant_step_index.unsqueeze(2).expand(-1, -1, hidden_dim)
        step_tokens = torch.gather(
            outputs, dim=1, index=step_indices
        )  # [batch_size, num_steps, hidden_dim]

        # 归一化向量以计算余弦相似度
        ans_emb_norm = F.normalize(ans_emb, dim=1)
        step_tokens_norm = F.normalize(step_tokens, dim=2)

        # 计算余弦相似度 [batch_size, num_steps]
        similarities = torch.bmm(step_tokens_norm, ans_emb_norm.unsqueeze(2)).squeeze(2)

        # 生成掩码并应用
        mask = assistant_step_index != 0  # 有效位置为 True
        similarities = similarities.masked_fill(~mask, -1e9)  # 填充位置设为极小值

        # 动态选择正样本（停止梯度传播）
        with torch.no_grad():
            pos_indices = similarities.argmax(dim=1)  # 只在有效位置中选择

        # 计算对比损失（温度系数 0.1）
        anchor_norm = F.normalize(anchor, dim=1)  # [B, H]
        similarity_anchor_step = torch.bmm(
            step_tokens_norm, anchor_norm.unsqueeze(2)
        ).squeeze(
            2
        )  # [B, S]
        similarity_anchor_step = similarity_anchor_step.masked_fill(~mask, -1e9)

        temperature = 0.1
        logits = similarity_anchor_step / temperature
        loss = F.cross_entropy(logits, pos_indices)

        return loss

    def compute_kl_loss(
        self, projected_inputs_embeds, q_and_step_index_list, thought_index
    ):
        """
        计算特殊 token 与问题 token 的 KL 散度损失

        参数:
            projected_inputs_embeds (Tensor): 形状 (batch_size, seq_len, 3584)
            q_and_step_index_list (Tensor): 形状 (batch_size, n)，第一个值为问题 token 的索引
            thought_index (Tensor): 形状 (batch_size, 4)，后两个值为特殊 token 的起始和结束索引

        返回:
            Tensor: 标量损失值
        """

        batch_size = projected_inputs_embeds.size(0)
        device = projected_inputs_embeds.device
        total_loss = torch.tensor(0.0, device=device)

        for i in range(batch_size):
            # 获取问题 token 的索引
            q_idx = q_and_step_index_list[i, 0].long()
            # 获取问题 token 的嵌入
            p_embed = projected_inputs_embeds[i, q_idx]  # (3584, )

            # 获取特殊 token 的起始和结束索引
            start_idx = thought_index[i, -2].long()
            end_idx = thought_index[i, -1].long()

            # 获取特殊 token 的嵌入
            q_embeds = projected_inputs_embeds[
                i, start_idx:end_idx
            ]  # (num_special, 3584)

            # 对问题 token 的嵌入进行 softmax 转换
            log_p = F.log_softmax(p_embed, dim=0)  # (3584, )
            p = torch.exp(log_p)  # (3584, )

            # 对特殊 token 的嵌入进行 log_softmax 转换
            log_q = F.log_softmax(q_embeds, dim=1)  # (num_special, 3584)

            # 扩展 p 到与 log_q 相同的形状
            p_expanded = p.expand(log_q.size(0), -1)  # (num_special, 3584)

            # 计算 KL 散度
            kl_elements = p_expanded * (log_p.expand_as(p_expanded) - log_q)
            kl_divs = kl_elements.sum(dim=1)  # (num_special, )

            # 取平均
            sample_loss = kl_divs.mean()
            total_loss += sample_loss

        # 取 batch 的平均损失
        total_loss = total_loss / batch_size

        return total_loss

    def get_inputs_embeds_for_base_model(
        self,
        assistant_input_ids,
        assistant_attention_mask,
        input_ids,
        inputs_embeds,
        thought_index,
        q_and_step_index_list,
        ans_ids,
        print_index=False,
    ):
        if self.num_thought_tokens == 0:
            if print_index:
                logger.info(
                    f"Number of thought tokens is zero, does not change the inputs embeds."
                )
            return inputs_embeds, torch.tensor(0.0, device=inputs_embeds.device)

        batch_size, seq_len, hidden_size = inputs_embeds.size()

        if self.model_type[0]:
            assistant_outputs = self.assistant_model(
                input_ids=assistant_input_ids,
                attention_mask=assistant_attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            assistant_hidden_states = assistant_outputs["hidden_states"][-1]
        else:
            assistant_input_embeds = self.base_model.get_input_embeddings()(
                assistant_input_ids
            )
            # assistant_hidden_states = assistant_input_embeds
            assistant_hidden_states = self.TransformerBlock(
                src=assistant_input_embeds,
                src_key_padding_mask=(assistant_attention_mask == 0).bool(),
            )
        assistant_hidden_states = assistant_hidden_states.to(
            self.projection.weight.dtype
        )
        projected_inputs_embeds = self.projection(assistant_hidden_states)
        if self.training:
            con_loss = self.contrastive_loss(
                projected_inputs_embeds, q_and_step_index_list, ans_ids
            )
            kl_loss = self.compute_kl_loss(
                projected_inputs_embeds, q_and_step_index_list, thought_index
            )
        else:
            con_loss = torch.tensor(0.0, device=self.device)
            kl_loss = torch.tensor(0.0, device=self.device)

        for b in range(batch_size):
            input_thought_start_idx = thought_index[b, 0].item()
            input_thought_end_idx = thought_index[b, 1].item()
            assistant_thought_start_idx = thought_index[b, 2].item()
            assistant_thought_end_idx = thought_index[b, 3].item()

            inputs_embeds[b, input_thought_start_idx:input_thought_end_idx] = (
                projected_inputs_embeds[
                    b, assistant_thought_start_idx:assistant_thought_end_idx
                ]
            )
            if print_index:
                raw_assistant_inputs = self.assistant_tokenizer.decode(
                    assistant_input_ids[
                        b, assistant_thought_start_idx:assistant_thought_end_idx
                    ]
                )
                if input_ids is not None:
                    raw_base_inputs = self.base_tokenizer.decode(
                        input_ids[b, input_thought_start_idx:input_thought_end_idx]
                    )
                else:
                    raw_base_inputs = f"Input IDs is None, embeddings from index {input_thought_start_idx} to {input_thought_end_idx}"
                logger.info(
                    f"Instance {b + 1}/{batch_size} - Embeddings from: <|start|>{raw_assistant_inputs}<|end|>"
                )
                logger.info(
                    f"Instance {b + 1}/{batch_size} - Embeddings to: <|start|>{raw_base_inputs}<|end|>"
                )

        return inputs_embeds, con_loss, kl_loss
