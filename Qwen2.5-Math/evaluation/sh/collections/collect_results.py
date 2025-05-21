import os
import json
import glob
import argparse
from collections import defaultdict
import pandas as pd
from transformers import AutoTokenizer
import wandb
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import threading
import matplotlib.pyplot as plt
import re

# Create a thread-local storage for tokenizer
thread_local = threading.local()

def get_tokenizer(model_name):
    """Get or create thread-local tokenizer"""
    if not hasattr(thread_local, 'tokenizer'):
        thread_local.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return thread_local.tokenizer

def normalize_model_name(path):
    """Extract and normalize model name from path"""
    parts = path.split('/')
    # First check for checkpoint pattern
    for part in parts[::-1]:
        if 'checkpoint' in part:
            idx = parts.index(part)
            model_name = parts[idx-1]
            checkpoint = part
            return f"{model_name}-{checkpoint}"
        # Add check for global_step pattern
        if 'global_step' in part:
            idx = parts.index(part)
            model_name = parts[idx-1]
            return f"{model_name}-{part}"
    
    # If no checkpoint or global_step found, use the last meaningful part and add checkpoint-final
    for part in reversed(parts):
        if any(x in part.lower() for x in ['llama', 'qwen', 'gpt', 'mistral']):
            return f"{part}-checkpoint-final"
    
    return "unknown_model"

def get_benchmark_name(path):
    """Extract benchmark name from path"""
    parts = path.split('/')
    # Look for common benchmark names in the path
    # for part in parts:
    #     if part.lower() in ['aime24', 'gsm8k', 'math500']:
    #         return part.lower()
    #TODO: potential bug for diff path
    return parts[-2]
    # return "unknown_benchmark"

def get_jsonl_path(metrics_file):
    """Get corresponding jsonl file path"""
    # Get the directory containing the metrics file
    metric_folder = os.path.dirname(metrics_file)
    
    # The JSONL file should be in the same directory with a .jsonl extension
    # and without the '_metrics' suffix
    base_name = os.path.basename(metrics_file).replace('_metrics.json', '')
    jsonl_file = os.path.join(metric_folder, f"{base_name}.jsonl")
    
    if not os.path.exists(jsonl_file):
        raise FileNotFoundError(f"JSONL file not found: {jsonl_file}")
    
    return jsonl_file

def calculate_avg_tokens_and_keywords(jsonl_path, tokenizer):
    """Calculate average tokens and keyword frequencies across all code elements"""
    if not os.path.exists(jsonl_path):
        print(f"Warning: JSONL file not found: {jsonl_path}")
        return 0, 0, 0, 0
    
    keywords = {"recheck", "rethink", "try again", "wait", "alternatively", "retry", "however"}
    total_tokens = 0
    total_keywords = 0
    total_correct_tokens = 0
    total_wrong_tokens = 0
    count = 0
    correct_count = 0
    wrong_count = 0
    
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if 'code' in data and isinstance(data['code'], list) and len(data['code']) > 0:
                    # Process all code samples instead of just the first one
                    sample_tokens = 0
                    sample_keywords = 0
                    
                    for code_sample in data['code']:
                        code_text = code_sample.lower()
                        tokens = len(tokenizer.encode(code_text))
                        sample_tokens += tokens
                        
                        # Count keywords
                        keyword_count = sum(code_text.count(keyword.lower()) for keyword in keywords)
                        sample_keywords += keyword_count
                    
                    # Calculate average per sample
                    avg_sample_tokens = sample_tokens / len(data['code'])
                    avg_sample_keywords = sample_keywords / len(data['code'])
                    
                    total_tokens += avg_sample_tokens
                    total_keywords += avg_sample_keywords
                    
                    # Separate tokens for correct and wrong answers
                    is_correct = data.get('score', [False])[0] if isinstance(data.get('score', []), list) else False
                    if is_correct:
                        total_correct_tokens += avg_sample_tokens
                        correct_count += 1
                    else:
                        total_wrong_tokens += avg_sample_tokens
                        wrong_count += 1
                    
                    count += 1
    except Exception as e:
        print(f"Error processing {jsonl_path}: {e}")
        return 0, 0, 0, 0
        
    avg_correct_tokens = total_correct_tokens / correct_count if correct_count > 0 else 0
    avg_wrong_tokens = total_wrong_tokens / wrong_count if wrong_count > 0 else 0
    
    return (total_tokens / count if count > 0 else 0,
            total_keywords / count if count > 0 else 0,
            avg_correct_tokens,
            avg_wrong_tokens)

def process_file(args):
    """Process a single metrics file"""
    metrics_file, model_name = args
    try:
        model_name_norm = normalize_model_name(metrics_file)
        benchmark = get_benchmark_name(metrics_file)
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            acc = metrics.get('acc', 0)
        
        jsonl_file = get_jsonl_path(metrics_file)
        tokenizer = get_tokenizer(model_name)
        avg_tokens, avg_keywords, avg_correct_tokens, avg_wrong_tokens = calculate_avg_tokens_and_keywords(jsonl_file, tokenizer)
        
        return model_name_norm, benchmark, {
            'acc': acc,
            'tokens': avg_tokens,
            'keywords': avg_keywords,
            'correct_tokens': avg_correct_tokens,
            'wrong_tokens': avg_wrong_tokens
        }
        
    except Exception as e:
        print(f"Error processing {metrics_file}: {e}")
        return None

def collect_results(base_dir, model_name, num_threads=8, temperature=None, prompt_type=None):
    # Initialize results storage
    results = defaultdict(lambda: defaultdict(dict))
    
    # Find all metrics.json files
    metrics_files = glob.glob(f"{base_dir}/**/test_*metrics.json", recursive=True)
    
    if temperature is not None:
        metrics_files = [f for f in metrics_files if f"t{temperature}" in f]
    
    if prompt_type is not None:
        metrics_files = [f for f in metrics_files if f"{prompt_type}" in f]
        
    # Create arguments for parallel processing
    process_args = [(f, model_name) for f in metrics_files]
    
    # Process files in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = list(tqdm(
            executor.map(process_file, process_args),
            total=len(metrics_files),
            desc="Processing files"
        ))
        
        # Collect results
        for result in futures:
            if result is not None:
                model_name, benchmark, metrics = result
                results[model_name][benchmark] = metrics
    
    return results

def create_summary(results):
    # Convert results to DataFrame
    rows = []
    for model, benchmarks in results.items():
        row = {'model': model}
        total_acc = 0
        total_tokens = 0
        count = 0
        
        for benchmark, metrics in benchmarks.items():
            row[f'{benchmark}_acc'] = metrics['acc']
            row[f'{benchmark}_tokens'] = metrics['tokens']
            total_acc += metrics['acc']
            total_tokens += metrics['tokens']
            count += 1
        
        if count > 0:
            row['avg_acc'] = total_acc / count
            row['avg_tokens'] = total_tokens / count
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort DataFrame by checkpoint/global_step number
    def get_step_number(model_name):
        if 'checkpoint-final' in model_name:
            return float('inf')
        # Check for checkpoint pattern
        checkpoint_match = re.search(r'checkpoint-(\d+)', model_name)
        if checkpoint_match:
            return int(checkpoint_match.group(1))
        # Check for global_step pattern
        global_step_match = re.search(r'global_step[_]?(\d+)', model_name)
        if global_step_match:
            return int(global_step_match.group(1))
        return float('inf')
    
    # Sort DataFrame based on step numbers
    df['sort_key'] = df['model'].apply(get_step_number)
    df = df.sort_values('sort_key')
    df = df.drop('sort_key', axis=1)
    
    return df

def sync_to_wandb(args, results, project_name, df, plot_dir, csv_path, temperature=None):
    """Sync results, CSV table and plots to wandb"""
    # Initialize wandb run
    runname = args.wandb_run_name
    if temperature is not None:
        runname = f"{runname}_t{temperature}"
    run = wandb.init(
        project=project_name,
        name=runname,
        reinit=True
    )
    
    # Log the CSV table as a wandb Table
    table = wandb.Table(dataframe=df)
    wandb.log({"results_table": table})
    
    # Also save the CSV file as an artifact
    artifact = wandb.Artifact('evaluation_results', type='dataset')
    artifact.add_file(csv_path)
    run.log_artifact(artifact)
    
    # Log plots
    if os.path.exists(plot_dir):
        for plot_file in os.listdir(plot_dir):
            if plot_file.endswith('_progress.png'):
                plot_path = os.path.join(plot_dir, plot_file)
                wandb.log({f"plots/{plot_file}": wandb.Image(plot_path)})
                
            if plot_file.endswith('_tokens_keywords.png'):
                plot_path = os.path.join(plot_dir, plot_file)
                wandb.log({f"plots/{plot_file}": wandb.Image(plot_path)})
                
            if plot_file.endswith('_acc_tokens.png'):
                plot_path = os.path.join(plot_dir, plot_file)
                wandb.log({f"plots/{plot_file}": wandb.Image(plot_path)})
                
            if plot_file.endswith('_acc_keywords.png'):
                plot_path = os.path.join(plot_dir, plot_file)
                wandb.log({f"plots/{plot_file}": wandb.Image(plot_path)})
                
            if plot_file.endswith('_correct_tokens.png'):
                plot_path = os.path.join(plot_dir, plot_file)
                wandb.log({f"plots/{plot_file}": wandb.Image(plot_path)})
                
            if plot_file.endswith('_wrong_tokens.png'):
                plot_path = os.path.join(plot_dir, plot_file)
                wandb.log({f"plots/{plot_file}": wandb.Image(plot_path)})
                
            if plot_file.endswith(f'_high_acc_{args.high_acc_threshold}.png'):
                plot_path = os.path.join(plot_dir, plot_file)
                wandb.log({f"plots/{plot_file}": wandb.Image(plot_path)})
    
    run.finish()

def sort_checkpoints(models):
    """Sort checkpoints numerically with final checkpoint at the end"""
    def get_checkpoint_num(model_name):
        if 'checkpoint-final' in model_name:
            return float('inf')
        # Check for checkpoint pattern
        checkpoint_match = re.search(r'checkpoint-(\d+)', model_name)
        if checkpoint_match:
            return int(checkpoint_match.group(1))
        # Check for global_step pattern
        global_step_match = re.search(r'global_step[_]?(\d+)', model_name)
        if global_step_match:
            return int(global_step_match.group(1))
        return float('inf')
    
    # Group models by base name (everything before checkpoint- or global_step)
    model_groups = defaultdict(list)
    for model in models:
        # Split on either checkpoint- or global_step
        base_name = re.split(r'(?:checkpoint-|global_step)', model)[0].rstrip('-')
        model_groups[base_name].append(model)
    
    # Sort each group's checkpoints
    sorted_models = []
    for base_name, checkpoints in model_groups.items():
        sorted_checkpoints = sorted(checkpoints, key=get_checkpoint_num)
        sorted_models.extend(sorted_checkpoints)
    
    return sorted_models

def plot_training_progress(results, output_dir, benchmarks=None, stop_on_zero_acc=False):
    """Plot training progress for each model series"""
    # Get all unique benchmarks
    all_benchmarks = set()
    for model_metrics in results.values():
        all_benchmarks.update(model_metrics.keys())
    all_benchmarks = sorted(list(all_benchmarks))
    
    # Filter benchmarks if specified
    if benchmarks:
        all_benchmarks = [b for b in all_benchmarks if b in benchmarks]
    
    # Group models by base name
    model_groups = defaultdict(list)
    for model in results.keys():
        base_name = re.split(r'(?:checkpoint-|global_step)', model)[0].rstrip('-')
        model_groups[base_name].append(model)
    
    # Create plots for each model group
    for base_name, models in model_groups.items():
        if len(models) <= 1:
            continue
            
        # Sort checkpoints
        models = sort_checkpoints(models)
        
        # If stop_on_zero_acc is True, find where to stop
        if stop_on_zero_acc:
            stop_idx = len(models)
            for idx, model in enumerate(models):
                avg_acc = 0
                metrics_count = 0
                for benchmark in all_benchmarks:
                    if benchmark in results[model]:
                        avg_acc += results[model][benchmark].get('acc', 0)
                        metrics_count += 1
                if metrics_count > 0:
                    avg_acc /= metrics_count
                    if avg_acc == 0:
                        stop_idx = idx
                        break
            models = models[:stop_idx]
            if len(models) == 0:
                continue

        # Extract checkpoint numbers for x-axis
        checkpoints = []
        for model in models:
            if 'checkpoint-final' in model:
                checkpoints.append('final')
            else:
                checkpoint_match = re.search(r'checkpoint-(\d+)', model)
                if checkpoint_match:
                    checkpoints.append(checkpoint_match.group(1))
                    continue
                global_step_match = re.search(r'global_step[_]?(\d+)', model)
                if global_step_match:
                    checkpoints.append(f'step{global_step_match.group(1)}')
                else:
                    checkpoints.append('unknown')

        # Create figures
        n_benchmarks = len(all_benchmarks) + 1  # +1 for average
        n_cols = 3
        n_rows = (n_benchmarks + n_cols - 1) // n_cols
        
        # Create three separate figures with the same layout
        for plot_type in ['acc_tokens', 'acc_keywords', 'tokens_keywords']:
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            fig.suptitle(f'Training Progress - {base_name}')
            axes = axes.flatten()
            
            # Plot average metrics first
            avg_metrics = defaultdict(list)
            for model in models:
                metrics = results[model]
                # 计算每个模型的平均值
                model_acc = []
                model_tokens = []
                model_keywords = []
                for benchmark in all_benchmarks:
                    if benchmark in metrics:
                        model_acc.append(metrics[benchmark].get('acc', 0))
                        model_tokens.append(metrics[benchmark].get('tokens', 0))
                        model_keywords.append(metrics[benchmark].get('keywords', 0))
                # 将每个模型的平均值添加到列表中
                avg_metrics['acc'].append(sum(model_acc) / len(model_acc) if model_acc else 0)
                avg_metrics['tokens'].append(sum(model_tokens) / len(model_tokens) if model_tokens else 0)
                avg_metrics['keywords'].append(sum(model_keywords) / len(model_keywords) if model_keywords else 0)
            
            # Plot first subplot (average)
            ax_twin = axes[0].twinx()
            
            if plot_type == 'acc_tokens':
                y1_data = avg_metrics['acc']
                y2_data = avg_metrics['tokens']
                y1_label, y2_label = 'Accuracy', 'Tokens'
                y1_color, y2_color = '#1f77b4', '#ff7f0e'
            elif plot_type == 'acc_keywords':
                y1_data = avg_metrics['acc']
                y2_data = avg_metrics['keywords']
                y1_label, y2_label = 'Accuracy', 'Keywords'
                y1_color, y2_color = '#1f77b4', '#2ca02c'
            else:  # tokens_keywords
                y1_data = avg_metrics['tokens']
                y2_data = avg_metrics['keywords']
                y1_label, y2_label = 'Tokens', 'Keywords'
                y1_color, y2_color = '#ff7f0e', '#2ca02c'
            
            line1 = axes[0].plot(range(len(checkpoints)), y1_data, marker='o', color=y1_color, label=y1_label)
            line2 = ax_twin.plot(range(len(checkpoints)), y2_data, marker='s', color=y2_color, label=y2_label)
            
            axes[0].set_title('Average Metrics')
            axes[0].set_xlabel('Checkpoint')
            axes[0].set_ylabel(y1_label, color=y1_color)
            ax_twin.set_ylabel(y2_label, color=y2_color)
            
            axes[0].set_xticks(range(len(checkpoints)))
            axes[0].set_xticklabels(checkpoints, rotation=45)
            axes[0].grid(True, alpha=0.3)
            
            # Add value annotations
            for i, (v1, v2) in enumerate(zip(y1_data, y2_data)):
                axes[0].annotate(f'{v1:.1f}', (i, v1), textcoords="offset points",
                               xytext=(0,10), ha='center', color=y1_color, fontsize=8)
                ax_twin.annotate(f'{v2:.2f}', (i, v2), textcoords="offset points",
                               xytext=(0,-15), ha='center', color=y2_color, fontsize=8)
            
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            axes[0].legend(lines, labels, loc='upper left')
            
            # Plot individual benchmarks
            for i, benchmark in enumerate(all_benchmarks, start=1):
                ax_twin = axes[i].twinx()
                
                y1_values = []
                y2_values = []
                for model in models:
                    metrics = results[model].get(benchmark, {})
                    if plot_type == 'acc_tokens':
                        y1_values.append(metrics.get('acc', 0))
                        y2_values.append(metrics.get('tokens', 0))
                    elif plot_type == 'acc_keywords':
                        y1_values.append(metrics.get('acc', 0))
                        y2_values.append(metrics.get('keywords', 0))
                    else:  # tokens_keywords
                        y1_values.append(metrics.get('tokens', 0))
                        y2_values.append(metrics.get('keywords', 0))
                
                line1 = axes[i].plot(range(len(checkpoints)), y1_values, marker='o', color=y1_color, label=y1_label)
                line2 = ax_twin.plot(range(len(checkpoints)), y2_values, marker='s', color=y2_color, label=y2_label)
                
                axes[i].set_title(benchmark)
                axes[i].set_xlabel('Checkpoint')
                axes[i].set_ylabel(y1_label, color=y1_color)
                ax_twin.set_ylabel(y2_label, color=y2_color)
                
                axes[i].set_xticks(range(len(checkpoints)))
                axes[i].set_xticklabels(checkpoints, rotation=45)
                axes[i].grid(True, alpha=0.3)
                
                for j, (v1, v2) in enumerate(zip(y1_values, y2_values)):
                    axes[i].annotate(f'{v1:.1f}', (j, v1), textcoords="offset points",
                                   xytext=(0,10), ha='center', color=y1_color, fontsize=8)
                    ax_twin.annotate(f'{v2:.2f}', (j, v2), textcoords="offset points",
                                   xytext=(0,-15), ha='center', color=y2_color, fontsize=8)
                
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                axes[i].legend(lines, labels, loc='upper left')
            
            # Remove empty subplots
            for i in range(len(all_benchmarks) + 1, len(axes)):
                fig.delaxes(axes[i])
            
            # Adjust layout and save
            fig.tight_layout()
            output_filename = os.path.join(output_dir, f'{base_name}_{plot_type}.png')
            
            if os.path.exists(output_filename):
                try:
                    os.remove(output_filename)
                except Exception as e:
                    print(f"Warning: Could not remove existing file {output_filename}: {e}")
            
            try:
                fig.savefig(output_filename)
                print(f"Saved plot to: {output_filename}")
            except Exception as e:
                print(f"Error saving plot: {e}")
            plt.close(fig)

    # Create two additional plots for correct/wrong tokens
    for base_name, models in model_groups.items():
        if len(models) <= 1:
            continue
            
        # Sort checkpoints
        models = sort_checkpoints(models)
        
        # Extract checkpoint numbers for x-axis
        checkpoints = []
        for model in models:
            if 'checkpoint-final' in model:
                checkpoints.append('final')
            else:
                checkpoint_match = re.search(r'checkpoint-(\d+)', model)
                if checkpoint_match:
                    checkpoints.append(checkpoint_match.group(1))
                    continue
                global_step_match = re.search(r'global_step[_]?(\d+)', model)
                if global_step_match:
                    checkpoints.append(f'step{global_step_match.group(1)}')
                else:
                    checkpoints.append('unknown')

        # Create figures for correct/wrong tokens
        for plot_type in ['correct_tokens', 'wrong_tokens']:
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            fig.suptitle(f'Training Progress - {base_name} - {"Correct" if plot_type == "correct_tokens" else "Wrong"} Answer Tokens')
            axes = axes.flatten()
            
            # Plot average metrics first
            avg_metrics = defaultdict(list)
            for model in models:
                metrics = results[model]
                model_acc = []
                model_tokens = []
                for benchmark in all_benchmarks:
                    if benchmark in metrics:
                        model_acc.append(metrics[benchmark].get('acc', 0))
                        model_tokens.append(metrics[benchmark].get(plot_type, 0))
                avg_metrics['acc'].append(sum(model_acc) / len(model_acc) if model_acc else 0)
                avg_metrics['tokens'].append(sum(model_tokens) / len(model_tokens) if model_tokens else 0)
            
            # Plot first subplot (average)
            ax_twin = axes[0].twinx()
            
            line1 = axes[0].plot(range(len(checkpoints)), avg_metrics['acc'], 
                               marker='o', color='#1f77b4', label='Accuracy')
            line2 = ax_twin.plot(range(len(checkpoints)), avg_metrics['tokens'], 
                               marker='s', color='#ff7f0e', 
                               label=f'{"Correct" if plot_type == "correct_tokens" else "Wrong"} Tokens')
            
            axes[0].set_title('Average Metrics')
            axes[0].set_xlabel('Checkpoint')
            axes[0].set_ylabel('Accuracy', color='#1f77b4')
            ax_twin.set_ylabel('Tokens', color='#ff7f0e')
            
            axes[0].set_xticks(range(len(checkpoints)))
            axes[0].set_xticklabels(checkpoints, rotation=45)
            axes[0].grid(True, alpha=0.3)
            
            # Add value annotations
            for i, (v1, v2) in enumerate(zip(avg_metrics['acc'], avg_metrics['tokens'])):
                axes[0].annotate(f'{v1:.1f}', (i, v1), textcoords="offset points",
                               xytext=(0,10), ha='center', color='#1f77b4', fontsize=8)
                ax_twin.annotate(f'{v2:.1f}', (i, v2), textcoords="offset points",
                               xytext=(0,-15), ha='center', color='#ff7f0e', fontsize=8)
            
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            axes[0].legend(lines, labels, loc='upper left')
            
            # Plot individual benchmarks
            for i, benchmark in enumerate(all_benchmarks, start=1):
                ax_twin = axes[i].twinx()
                
                acc_values = []
                token_values = []
                for model in models:
                    metrics = results[model].get(benchmark, {})
                    acc_values.append(metrics.get('acc', 0))
                    token_values.append(metrics.get(plot_type, 0))
                
                line1 = axes[i].plot(range(len(checkpoints)), acc_values, 
                                   marker='o', color='#1f77b4', label='Accuracy')
                line2 = ax_twin.plot(range(len(checkpoints)), token_values, 
                                   marker='s', color='#ff7f0e', 
                                   label=f'{"Correct" if plot_type == "correct_tokens" else "Wrong"} Tokens')
                
                axes[i].set_title(benchmark)
                axes[i].set_xlabel('Checkpoint')
                axes[i].set_ylabel('Accuracy', color='#1f77b4')
                ax_twin.set_ylabel('Tokens', color='#ff7f0e')
                
                axes[i].set_xticks(range(len(checkpoints)))
                axes[i].set_xticklabels(checkpoints, rotation=45)
                axes[i].grid(True, alpha=0.3)
                
                for j, (v1, v2) in enumerate(zip(acc_values, token_values)):
                    axes[i].annotate(f'{v1:.1f}', (j, v1), textcoords="offset points",
                                   xytext=(0,10), ha='center', color='#1f77b4', fontsize=8)
                    ax_twin.annotate(f'{v2:.1f}', (j, v2), textcoords="offset points",
                                   xytext=(0,-15), ha='center', color='#ff7f0e', fontsize=8)
                
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                axes[i].legend(lines, labels, loc='upper left')
            
            # Remove empty subplots
            for i in range(len(all_benchmarks) + 1, len(axes)):
                fig.delaxes(axes[i])
            
            # Adjust layout and save
            fig.tight_layout()
            output_filename = os.path.join(output_dir, f'{base_name}_{plot_type}.png')
            
            if os.path.exists(output_filename):
                try:
                    os.remove(output_filename)
                except Exception as e:
                    print(f"Warning: Could not remove existing file {output_filename}: {e}")
            
            try:
                fig.savefig(output_filename)
                print(f"Saved plot to: {output_filename}")
            except Exception as e:
                print(f"Error saving plot: {e}")
            plt.close(fig)

def plot_high_acc_progress(results, output_dir, acc_threshold=78.0, benchmarks=None):
    """Plot training progress for checkpoints with average accuracy above threshold"""
    # Get all unique benchmarks
    all_benchmarks = set()
    for model_metrics in results.values():
        all_benchmarks.update(model_metrics.keys())
    all_benchmarks = sorted(list(all_benchmarks))
    
    # Filter benchmarks if specified
    if benchmarks:
        all_benchmarks = [b for b in all_benchmarks if b in benchmarks]
    
    # Group models by base name
    model_groups = defaultdict(list)
    for model in results.keys():
        base_name = re.split(r'(?:checkpoint-|global_step)', model)[0].rstrip('-')
        model_groups[base_name].append(model)
    
    # Create plots for each model group
    for base_name, models in model_groups.items():
        if len(models) <= 1:
            continue
            
        # Sort checkpoints
        models = sort_checkpoints(models)
        
        # Filter models by average accuracy threshold
        filtered_models = []
        filtered_checkpoints = []
        
        for model in models:
            # Calculate average accuracy across all benchmarks
            total_acc = 0
            count = 0
            for benchmark in all_benchmarks:
                if benchmark in results[model]:
                    total_acc += results[model][benchmark].get('acc', 0)
                    count += 1
            
            avg_acc = total_acc / count if count > 0 else 0
            
            # Only include models with average accuracy above threshold
            if avg_acc > acc_threshold:
                filtered_models.append(model)
                
                # Extract checkpoint number for x-axis label
                if 'checkpoint-final' in model:
                    filtered_checkpoints.append('final')
                else:
                    checkpoint_match = re.search(r'checkpoint-(\d+)', model)
                    if checkpoint_match:
                        filtered_checkpoints.append(checkpoint_match.group(1))
                        continue
                    global_step_match = re.search(r'global_step[_]?(\d+)', model)
                    if global_step_match:
                        filtered_checkpoints.append(f'step{global_step_match.group(1)}')
                    else:
                        filtered_checkpoints.append('unknown')
        
        # Skip if no models meet the threshold
        if len(filtered_models) <= 1:
            print(f"No models for {base_name} meet the accuracy threshold of {acc_threshold}%")
            continue

        # Create figure layout
        n_benchmarks = len(all_benchmarks) + 1  # +1 for average
        n_cols = 3
        n_rows = (n_benchmarks + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        fig.suptitle(f'High Accuracy Progress (>{acc_threshold}%) - {base_name}')
        axes = axes.flatten()
        
        # Plot average metrics first
        avg_metrics = defaultdict(list)
        for model in filtered_models:
            metrics = results[model]
            # Calculate average for each model
            model_acc = []
            model_tokens = []
            for benchmark in all_benchmarks:
                if benchmark in metrics:
                    model_acc.append(metrics[benchmark].get('acc', 0))
                    model_tokens.append(metrics[benchmark].get('tokens', 0))
            # Add each model's average to the list
            avg_metrics['acc'].append(sum(model_acc) / len(model_acc) if model_acc else 0)
            avg_metrics['tokens'].append(sum(model_tokens) / len(model_tokens) if model_tokens else 0)
        
        # Plot first subplot (average)
        ax_twin = axes[0].twinx()
        
        line1 = axes[0].plot(range(len(filtered_checkpoints)), avg_metrics['acc'], 
                           marker='o', color='#1f77b4', label='Accuracy')
        line2 = ax_twin.plot(range(len(filtered_checkpoints)), avg_metrics['tokens'], 
                           marker='s', color='#ff7f0e', label='Tokens')
        
        axes[0].set_title('Average Metrics')
        axes[0].set_xlabel('Checkpoint')
        axes[0].set_ylabel('Accuracy', color='#1f77b4')
        ax_twin.set_ylabel('Tokens', color='#ff7f0e')
        
        axes[0].set_xticks(range(len(filtered_checkpoints)))
        axes[0].set_xticklabels(filtered_checkpoints, rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # Add value annotations
        for i, (v1, v2) in enumerate(zip(avg_metrics['acc'], avg_metrics['tokens'])):
            axes[0].annotate(f'{v1:.1f}', (i, v1), textcoords="offset points",
                           xytext=(0,10), ha='center', color='#1f77b4', fontsize=8)
            ax_twin.annotate(f'{v2:.2f}', (i, v2), textcoords="offset points",
                           xytext=(0,-15), ha='center', color='#ff7f0e', fontsize=8)
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        axes[0].legend(lines, labels, loc='upper left')
        
        # Plot individual benchmarks
        for i, benchmark in enumerate(all_benchmarks, start=1):
            ax_twin = axes[i].twinx()
            
            acc_values = []
            token_values = []
            for model in filtered_models:
                metrics = results[model].get(benchmark, {})
                acc_values.append(metrics.get('acc', 0))
                token_values.append(metrics.get('tokens', 0))
            
            line1 = axes[i].plot(range(len(filtered_checkpoints)), acc_values, 
                               marker='o', color='#1f77b4', label='Accuracy')
            line2 = ax_twin.plot(range(len(filtered_checkpoints)), token_values, 
                               marker='s', color='#ff7f0e', label='Tokens')
            
            axes[i].set_title(benchmark)
            axes[i].set_xlabel('Checkpoint')
            axes[i].set_ylabel('Accuracy', color='#1f77b4')
            ax_twin.set_ylabel('Tokens', color='#ff7f0e')
            
            axes[i].set_xticks(range(len(filtered_checkpoints)))
            axes[i].set_xticklabels(filtered_checkpoints, rotation=45)
            axes[i].grid(True, alpha=0.3)
            
            for j, (v1, v2) in enumerate(zip(acc_values, token_values)):
                axes[i].annotate(f'{v1:.1f}', (j, v1), textcoords="offset points",
                               xytext=(0,10), ha='center', color='#1f77b4', fontsize=8)
                ax_twin.annotate(f'{v2:.2f}', (j, v2), textcoords="offset points",
                               xytext=(0,-15), ha='center', color='#ff7f0e', fontsize=8)
            
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            axes[i].legend(lines, labels, loc='upper left')
        
        # Remove empty subplots
        for i in range(len(all_benchmarks) + 1, len(axes)):
            fig.delaxes(axes[i])
        
        # Adjust layout and save
        fig.tight_layout()
        output_filename = os.path.join(output_dir, f'{base_name}_high_acc_{acc_threshold}.png')
        
        if os.path.exists(output_filename):
            try:
                os.remove(output_filename)
            except Exception as e:
                print(f"Warning: Could not remove existing file {output_filename}: {e}")
        
        try:
            fig.savefig(output_filename)
            print(f"Saved high accuracy plot to: {output_filename}")
        except Exception as e:
            print(f"Error saving plot: {e}")
        plt.close(fig)

def main(args):
    base_dir = args.base_dir
    model_name = args.model_name
    
    # Parse benchmarks if specified
    benchmarks = None
    if args.benchmarks:
        benchmarks = set(args.benchmarks.split(','))
    
    # Collect results
    print("Collecting results...")
    results = collect_results(base_dir, model_name, args.num_threads, args.temperature, args.prompt_type)
    
    # Filter results if benchmarks specified
    if benchmarks:
        filtered_results = defaultdict(lambda: defaultdict(dict))
        for model, model_results in results.items():
            for benchmark, metrics in model_results.items():
                if benchmark in benchmarks:
                    filtered_results[model][benchmark] = metrics
        results = filtered_results
    
    # Create summary DataFrame
    print("\nCreating summary...")
    df = create_summary(results)
    print("\nResults summary:")
    print(df)
    
    # Plot training progress
    print("\nCreating training progress plots...")
    plot_training_progress(results, args.plot_dir, benchmarks, args.stop_on_zero_acc)
    
    # Plot high accuracy progress
    print("\nCreating high accuracy progress plots...")
    plot_high_acc_progress(results, args.plot_dir, args.high_acc_threshold, benchmarks)
    
    # Save to CSV
    output_file = args.output_path
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # Sync to wandb if enabled
    if args.use_wandb:
        print("\nSyncing to wandb...")
        if args.wandb_api_key:
            wandb.login(key=args.wandb_api_key)
        sync_to_wandb(args, results, args.wandb_project, df, args.plot_dir, args.output_path, args.temperature)
        print("Wandb sync completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="outputs/project/reasonshort/weiliu/cot/output")
    parser.add_argument("--model_name", type=str, default="Qwen-math-7B-S100-qwq-fs-7k8-8192len-5e-6-rope10-bsz64")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--plot_dir", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="math-eval-results")
    parser.add_argument("--wandb_api_key", type=str, default="")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--num_threads", type=int, default=8)
    parser.add_argument("--benchmarks", type=str, 
                       default="gsm8k,math,minerva_math,olympiadbench,college_math,aime24,amc23",
                       help="Comma-separated list of benchmarks to include")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--stop_on_zero_acc", action="store_true", 
                       help="Stop plotting after encountering zero accuracy")
    parser.add_argument("--high_acc_threshold", type=float, default=78.0,
                       help="Threshold for high accuracy plots (only include checkpoints above this value)")
    parser.add_argument("--prompt_type", type=str, default=None,
                       help="Prompt type to include in the results")
    
    args = parser.parse_args()
    
    if args.temperature == -1:
        args.temperature = None
    
    if args.output_path is None:
        args.output_path = os.path.join(args.base_dir, "eval_results.csv")
    
    if args.plot_dir is None:
        args.plot_dir = os.path.join(args.base_dir, "plots")
        
    if not os.path.exists(args.plot_dir):
        os.makedirs(args.plot_dir, exist_ok=True)
        
    main(args)
