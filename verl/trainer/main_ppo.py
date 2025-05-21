# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
from verl.utils.reward_score import gsm8k, math
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.reward_score import kk
from verl.utils.reward_score import simplelr_math
from verl.utils.reward_score import length_penalty
from verl.utils.reward_score import simplelr_math_qwen
from verl.utils.reward_score import hf_math_verifier
from verl.utils.reward_score.constants import get_length_threshold

def _default_compute_score(data_source, solution_str, ground_truth, format_reward=True):
    if data_source == 'openai/gsm8k':
        return gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ['lighteval/MATH', 'DigitalLearningGmbH/MATH-lighteval']:
        return math.compute_score(solution_str, ground_truth)
    elif "kk" in data_source:
        return kk.compute_score(solution_str, ground_truth)
    elif "simplelr" in data_source or data_source in ["math_level_3-5", "GAIR/LIMO", "math_level_1-2", "gsm8k_r1", "deepscaler"]:
        # return simplelr_math.compute_score(solution_str, ground_truth)
        if format_reward:
            return hf_math_verifier.compute_score(solution_str, ground_truth)
        else:
            return hf_math_verifier.compute_score_no_format(solution_str, ground_truth)
        # return simplelr_math_qwen.compute_score(solution_str, ground_truth)
    else:
        raise NotImplementedError


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.format_reward = kwargs.get('format_reward', True)
        
    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        correctness_tensor = torch.zeros(len(data), dtype=torch.float32)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            score = self.compute_score(
                data_source=data_source,
                solution_str=sequences_str,
                ground_truth=ground_truth,
                format_reward=self.format_reward
            )
            
            if isinstance(score, dict):
                reward_tensor[i, valid_response_length - 1] = score['score']
                correctness_tensor[i] = score['correctness']
            else:
                reward_tensor[i, valid_response_length - 1] = score
                correctness_tensor[i] = 100.0

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)
        
        return {"reward_tensor": reward_tensor, "correctness_tensor": correctness_tensor}
    
class LengthyRewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, length_penalty=None, len_coef=None, incorrect_len_coef=None, acc_weight=None, **kwargs) -> None:
        
        """
            If length_penalty is "detect", the reward manager is only used for recording
            the pass@k and accuracy under different length constraints
            It will pass these stats to the length controller
        """
        
        print(f"Length penalty is enabled")
        print(f"Length penalty method: {length_penalty}")
        print(f"Length penalty coefficient: {len_coef}")
        print("Compute score: ", compute_score)

        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or _default_compute_score
        self.length_penalty = length_penalty
        self.length_coef = len_coef
        self.incorrect_len_coef = incorrect_len_coef
        self.acc_weight = acc_weight
        
        self.length_reward_scalar = kwargs.get('length_reward_scalar', None)
        self.target_length = kwargs.get('target_length', None)
        self.length_reward_type = kwargs.get('length_reward_type', "reward")
        
        self.expected_n_correct = kwargs.get('expected_n_correct', 1)
        self.lower_coverage_ratio = kwargs.get('lower_coverage_ratio', -1)
        
        lower_bound = kwargs.get('lower_bound', None)
        upper_bound = kwargs.get('upper_bound', None)
        interval = kwargs.get('interval', None)
        
        self.round_method = kwargs.get('round_method', 'floor')
        
        assert lower_bound is not None and upper_bound is not None and interval is not None, "lower_bound, upper_bound, and interval must be provided"
        
        self.length_thresholds = get_length_threshold(lower_bound, upper_bound, interval)
        
        self.format_reward = kwargs.get('format_reward', True)
        
    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        correctness_tensor = torch.zeros(len(data), dtype=torch.float32)
        
        score_scalar_tensor = torch.zeros(len(data), dtype=torch.float32)
        detecting_score_scalar_tensor = torch.zeros(len(data), dtype=torch.float32)
        detecting_correctness_tensor = torch.zeros(len(data), dtype=torch.float32)
        
        original_score_scalar_tensor = torch.zeros(len(data), dtype=torch.float32)

        already_print_data_sources = {}
        
        # data = self.length_penalty(data)

        all_score_dict_lst = []
        all_detecting_score_dict_lst = []
        
        length_to_detect = None
        if "length_to_detect" in data.meta_info.keys():
            print(f"Enable length detection with length: {data.meta_info['length_to_detect']}")
            length_to_detect = data.meta_info["length_to_detect"]
        
        for i in range(len(data)):
            
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch['data_source']

            score_dict = self.compute_score(
                data_source=data_source,
                solution_str=sequences_str,
                ground_truth=ground_truth,
                format_reward=self.format_reward
            )
            
            all_score_dict_lst.append(score_dict)
            
            if length_to_detect is not None:
                detecting_valid_response_length = min(valid_response_length, length_to_detect)
                detecting_valid_response_ids = response_ids[:detecting_valid_response_length]
                detecting_sequences = torch.cat((valid_prompt_ids, detecting_valid_response_ids))
                detecting_sequences_str = self.tokenizer.decode(detecting_sequences)
                
                detecting_score_dict = self.compute_score(
                    data_source=data_source,
                    solution_str=detecting_sequences_str,
                    ground_truth=ground_truth,
                    format_reward=self.format_reward
                )
                
                all_detecting_score_dict_lst.append(detecting_score_dict)
        
        data.batch['correctness'] = correctness_tensor
        
        all_score_lst = self.length_penalty(data, all_score_dict_lst,
                                            len_coef=self.length_coef,
                                            incorrect_len_coef=self.incorrect_len_coef,
                                            acc_weight=self.acc_weight,
                                            length_reward_scalar=self.length_reward_scalar,
                                            target_length=self.target_length,
                                            length_reward_type=self.length_reward_type,
                                            expected_n_correct=self.expected_n_correct,
                                            lower_coverage_ratio=self.lower_coverage_ratio,
                                            length_thresholds=self.length_thresholds,
                                            round_method=self.round_method)
        
        for i in range(len(data)):
            
            processed_score: float = all_score_lst[i]
            original_score = all_score_dict_lst[i]['score']
            correctness = all_score_dict_lst[i]['correctness']
            
            original_score_scalar_tensor[i] = original_score
            reward_tensor[i, valid_response_length - 1] = processed_score
            correctness_tensor[i] = correctness
    
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)
                
            if length_to_detect is not None:
                detecting_score = all_detecting_score_dict_lst[i]['score']
                detecting_correctness = all_detecting_score_dict_lst[i]['correctness']
                
                detecting_score_scalar_tensor[i] = detecting_score
                detecting_correctness_tensor[i] = detecting_correctness
        
        if length_to_detect is not None:
            return {"reward_tensor": reward_tensor, "correctness_tensor": correctness_tensor, "original_score": original_score_scalar_tensor,
                    "detecting_score_scalar_tensor": detecting_score_scalar_tensor, "detecting_correctness_tensor": detecting_correctness_tensor}
        else:
            return {"reward_tensor": reward_tensor, "correctness_tensor": correctness_tensor, "original_score": original_score_scalar_tensor}

import ray
import hydra


@hydra.main(config_path='config', config_name='lengthy_ppo_trainer', version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config, compute_score=None):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config, compute_score))


@ray.remote
def main_task(config, compute_score=None):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id
        
    if hasattr(config, 'length_penalty'):
        if config.length_penalty.enable:
            print("Length penalty is enabled")
            print(f"Length penalty method: {config.length_penalty.method}")
            print(f"Length penalty coefficient: {config.length_penalty.length_coef}")
            
            if config.length_penalty.method == 'larse':
                length_penalty_fn = length_penalty.compute_length_reward_larse
            elif config.length_penalty.method == 'larse-d':
                length_penalty_fn = length_penalty.compute_length_reward_larse_d
            elif config.length_penalty.method == 'larse-de':
                length_penalty_fn = length_penalty.compute_length_reward_larse_de
            else:
                raise NotImplementedError
            
            # compute_score_fn = simplelr_math.compute_score_with_length
            
            reward_fn = LengthyRewardManager(
                tokenizer=tokenizer, 
                num_examine=0, 
                length_penalty=length_penalty_fn,
                len_coef=config.length_penalty.length_coef,
                incorrect_len_coef=config.length_penalty.incorrect_len_coef,
                acc_weight=config.length_penalty.acc_weight,
                length_reward_scalar=config.length_penalty.length_reward_scalar,
                target_length=config.length_penalty.target_length,
                length_reward_type=config.length_penalty.length_reward_type,
                expected_n_correct=config.length_penalty.expected_n_correct,
                lower_coverage_ratio=config.length_penalty.lower_coverage_ratio,
                format_reward=config.trainer.format_reward,
                lower_bound=config.length_penalty.lower_bound,
                upper_bound=config.length_penalty.upper_bound,
                interval=config.length_penalty.interval,
                round_method=config.length_penalty.round_method
            )
        else:
            reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, compute_score=compute_score, format_reward=config.trainer.format_reward)
    else:
        reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0, compute_score=compute_score, format_reward=config.trainer.format_reward)

    # Note that we always use function-based RM for validation
    if hasattr(config, 'length_penalty'):
        if config.length_penalty.val_enable:
            val_reward_fn = LengthyRewardManager(tokenizer=tokenizer, 
                                                 num_examine=1, 
                                                 length_penalty=length_penalty_fn,
                                                 len_coef=config.length_penalty.length_coef,
                                                 incorrect_len_coef=config.length_penalty.incorrect_len_coef,
                                                 acc_weight=config.length_penalty.acc_weight,
                                                 length_reward_scalar=config.length_penalty.length_reward_scalar,
                                                 target_length=config.length_penalty.target_length,
                                                 length_reward_type=config.length_penalty.length_reward_type,
                                                 expected_n_correct=config.length_penalty.expected_n_correct,
                                                 lower_coverage_ratio=config.length_penalty.lower_coverage_ratio,
                                                 format_reward=config.trainer.format_reward,
                                                 lower_bound=config.length_penalty.lower_bound,
                                                 upper_bound=config.length_penalty.upper_bound,
                                                 interval=config.length_penalty.interval,
                                                 round_method=config.length_penalty.round_method)
        else:
            val_reward_fn = RewardManager(tokenizer=tokenizer, 
                                          num_examine=1, 
                                          compute_score=compute_score,
                                          format_reward=config.trainer.format_reward)
    else:
        val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1, compute_score=compute_score, format_reward=config.trainer.format_reward)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn)
    trainer.init_workers()
    trainer.fit()

if __name__ == '__main__':
    main()
