
from typing import Any


class LengthController:
    def __init__(self, ctrl_config):
        
        self.ctrl_config = ctrl_config
        
        self.init_length = ctrl_config.init_length
        self.current_length = ctrl_config.init_length
        self.max_length = ctrl_config.max_length
        self.min_length = ctrl_config.min_length
        self.start_step = ctrl_config.start_step
        self.length_strategy = ctrl_config.length_strategy
        
        self.acc_good_clip_ratio = 0
        self.current_clip_ratio = None  # it is used to resume from the last clip ratio
        self.target_acc_times = ctrl_config.target_acc_times
        self.target_clip_ratio = ctrl_config.target_clip_ratio
        
        self.target_passk_k = ctrl_config.target_passk_k
        self.target_comparison_num = ctrl_config.target_comparison_num
        
        self.interval = ctrl_config.interval
        self.reduction = ctrl_config.reduction
        
        self.passk_distribution_metrics = {}
    
    def interval_reduction(self, step: int):
        
        if ((step - self.start_step) % self.interval == 0) and step >= self.start_step:
            original_length = self.current_length
            self.current_length = max(self.current_length - self.reduction, self.min_length)
            print(f"Original length: {original_length}, New length: {self.current_length}")
            
    def monitor_clip_ratio(self, clip_ratio: float, step: int = None):
        
        assert clip_ratio is not None, "clip_ratio must be provided"
        assert self.target_clip_ratio is not None, "target_clip_ratio must be provided"
        assert self.target_acc_times is not None, "target_acc_times must be provided"
        
        if clip_ratio == -1:
            if self.current_clip_ratio is None:
                return
            else:
                clip_ratio = self.current_clip_ratio
        
        if step >= self.start_step:
            if clip_ratio <= self.target_clip_ratio:
                self.acc_good_clip_ratio += 1
                print(f"Clip ratio is satisfied: {clip_ratio} <= {self.target_clip_ratio} || acc_good_clip_ratio: {self.acc_good_clip_ratio}")
            else:
                self.acc_good_clip_ratio = 0
                print(f"Clip ratio is not satisfied: {clip_ratio} > {self.target_clip_ratio} || acc_good_clip_ratio: {self.acc_good_clip_ratio}")
                
            if self.acc_good_clip_ratio >= self.target_acc_times:
                current_length = self.current_length
                self.current_length = max(self.current_length - self.reduction, self.min_length)
                print(f"Clip ratio is satisfied for {self.target_acc_times} times. Reduce length from {current_length} to {self.current_length}")
                self.acc_good_clip_ratio = 0
        else:
            print(f"Step {step} is less than start_step {self.start_step}, skip length reduction")
        
        self.current_clip_ratio = clip_ratio
        
    def monitor_passk_distribution(self, passk_distribution_metrics: dict, step: int = None):
        
        """
            key name:
                metrics[f'detecting_passk_comparison/regular_pass_at_{k}_ratio'] = pass_k_ratio_regular
                metrics[f'detecting_passk_comparison/detecting_pass_at_{k}_ratio'] = pass_k_ratio_detecting
                metrics[f'detecting_passk_comparison/difference_pass_at_{k}_ratio'] = ratio_difference
        """
        
        assert passk_distribution_metrics is not None, "passk_distribution_metrics must be provided"
        
        if self.passk_distribution_metrics == {} and passk_distribution_metrics == {}:
            print(f"WARNING: Pass@k distribution metrics is empty at step {step}. Use self.passk_distribution_metrics instead. This may be caused by resuming from checkpoint.")
            return
        
        if step >= self.start_step:
            
            if passk_distribution_metrics != {}:
                self.passk_distribution_metrics = passk_distribution_metrics
            else:
                print(f"WARNING: Pass@k distribution metrics is empty at step {step}. Use self.passk_distribution_metrics instead. This may be caused by resuming from checkpoint.")
                passk_distribution_metrics = self.passk_distribution_metrics
            
            difference_pass_at_k_ratio = passk_distribution_metrics[f'detecting_passk_comparison/difference_pass_at_{self.target_passk_k}_ratio']
            
            if abs(difference_pass_at_k_ratio) <= self.target_comparison_num or difference_pass_at_k_ratio >= 0:
                self.current_length = max(self.current_length - self.reduction, self.min_length)
                print(f"Pass@K distribution is satisfied for {self.target_comparison_num} times. Reduce length from {self.current_length} to {self.current_length}")
            else:
                print(f"Pass@K distribution is not satisfied for {self.target_comparison_num} times. Current length: {self.current_length}")
            
    def __call__(self, global_step: int, clip_ratio: float, passk_distribution_metrics: dict) -> int:

        if self.length_strategy == "interval_reduction":
            self.interval_reduction(global_step)
        elif self.length_strategy == "constant":
            self.current_length = self.init_length
        elif self.length_strategy == "monitor_clip_ratio":
            self.monitor_clip_ratio(clip_ratio, global_step)
        elif self.length_strategy == "monitor_passk_distribution":
            self.monitor_passk_distribution(passk_distribution_metrics, global_step)
        else:
            raise ValueError(f"Invalid length strategy: {self.length_strategy}")
        
        return self.current_length
