# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Arc Institute. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Michael Poli. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Stanford University. All rights reserved
# SPDX-License-Identifier: LicenseRef-Apache2
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
import torch
import argparse
from pathlib import Path
from typing import List, Optional
import csv
import os
import random
import logging
from bionemo.testing.torch import check_fp8_support
from megatron.core.inference.common_inference_params import CommonInferenceParams
from nemo.collections.llm.inference import generate

from bionemo.core.data.load import load

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse arguments for Evo2 inference."""
    ap = argparse.ArgumentParser(description='Generate DNA sequences from prompts')
    ap.add_argument(
        "--prompt",
        type=str,
        default="ATCG",
        help="Prompt to generate text from Evo2. Single DNA prompt sequence.",
    )
    ap.add_argument(
        "--ckpt-name", type=str, default="evo2/1b-8k-bf16:1.0", help="Name of checkpoint directory containing pre-trained Evo2 model."
    )
    ap.add_argument("--temperature", type=float, default=1.0, help="Temperature during sampling for generation.")
    ap.add_argument("--top-k", type=int, default=4, help="Top K during sampling for generation.")
    ap.add_argument("--top-p", type=float, default=0.0, help="Top P during sampling for generation.")
    ap.add_argument("--num-tokens", type=int, default=1000, help="Number of tokens to generate.")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for generation.")
    ap.add_argument(
        "--output-file",
        type=str,
        default="generated_sequences.csv",
        help="Output CSV file path",
    )
    ap.add_argument('--num-generations', type=int, default=1, help="Number of generations to perform")

    return ap.parse_args()


################################################################################
# Batch generation
################################################################################
def get_trainer(pipeline_parallel=1):
    import nemo.lightning as nl

    fp8 = True
    full_fp8 = False
    return nl.Trainer(
        accelerator="gpu",
        devices=pipeline_parallel,
        strategy=nl.MegatronStrategy(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=pipeline_parallel,
            context_parallel_size=1,
            pipeline_dtype=torch.bfloat16,
            ckpt_load_optimizer=False,
            ckpt_save_optimizer=False,
            ckpt_async_save=False,
            save_ckpt_format="torch_dist",
            ckpt_load_strictness="log_all",
        ),
        log_every_n_steps=1,
        limit_val_batches=10,
        num_sanity_val_steps=0,
        plugins=nl.MegatronMixedPrecision(
            precision="bf16-mixed",
            params_dtype=torch.bfloat16,
            # Only use FP8 in this plugin when using full FP8 precision and FP8.
            #   Otherwise use vortex_style_fp8 in the model config.
            fp8="hybrid" if fp8 and full_fp8 else None,
            fp8_amax_history_len=16 if fp8 and full_fp8 else 1,
            fp8_amax_compute_algo="max" if fp8 and full_fp8 else "most_recent",
        ),
    )


def get_model_and_tokenizer_raw(ckpt_dir_or_name: Path | str, **kwargs):
    """
    Load a model and tokenizer from a checkpoint directory or name. If you supply a Path argument then we assume that
    the path is already a checkpoint directory, otherwise we load the checkpoint from NGC or PBSS depending on
    the environment variable BIONEMO_DATA_SOURCE.
    """
    trainer = get_trainer()
    from bionemo.core.data.load import load

    if isinstance(ckpt_dir_or_name, Path):
        ckpt_dir: Path = ckpt_dir_or_name
    else:
        ckpt_dir: Path = load(ckpt_dir_or_name)
    from nemo.collections.llm import inference

    inference_wrapped_model, mcore_tokenizer = inference.setup_model_and_tokenizer(
        path=ckpt_dir,
        trainer=trainer,
        params_dtype=torch.bfloat16,
        inference_batch_times_seqlen_threshold=8192,  # TODO
        inference_max_seq_length=8192,  # TODO
        recompute_granularity=None,
        recompute_num_layers=None,
        recompute_method=None,
        **kwargs,
    )
    return inference_wrapped_model, mcore_tokenizer


def save_sequences_csv(sequences: List[str], output_file: str, hyperparameters: dict, generation_round: int = 1):
    """Save generated sequences and hyperparameters to a CSV file."""
    
    # Check if file exists and if we need to write header
    write_header = not os.path.exists(output_file) or generation_round == 1
    mode = 'w' if generation_round == 1 else 'a'
    
    with open(output_file, mode, newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['Sequence', 'Generation_Round', 'Prompt', 'Model', 'Temperature', 'Top_K', 'Top_P', 'Num_Tokens', 'Seed'])

        for seq in sequences:
            # Extract individual hyperparameters for cleaner CSV format
            row = [
                seq,
                generation_round,
                hyperparameters.get('prompt', '')[:50] + '...' if len(hyperparameters.get('prompt', '')) > 50 else hyperparameters.get('prompt', ''),
                hyperparameters.get('model_name', ''),
                hyperparameters.get('temperature', ''),
                hyperparameters.get('top_k', ''),
                hyperparameters.get('top_p', ''),
                hyperparameters.get('n_tokens', ''),
                hyperparameters.get('seed', '')
            ]
            writer.writerow(row)





def predict(ckpt_name="evo2/1b-8k-bf16:1.0", num_tokens=1000, prompt="ATCG", temperature=1.0, 
           top_k=1, top_p=0.0, seed=None, output_file="generated_sequences.csv", 
           num_generations=1):
    """Run prediction with the model and save results to CSV."""
    
    logger.info(f"Predicting with checkpoint: {ckpt_name}")
    is_fp8_supported, compute_capability, device_info = check_fp8_support(torch.cuda.current_device())
    vortex_style_fp8 = is_fp8_supported and "bf16" not in ckpt_name
    logger.info(f"Vortex style FP8: {vortex_style_fp8}")
    
    # Load model and tokenizer once
    inference_wrapped_model, mcore_tokenizer = get_model_and_tokenizer_raw(ckpt_name, vortex_style_fp8=vortex_style_fp8)
    
    # Prepare hyperparameters dictionary for saving
    hyperparameters = {
        'model_name': ckpt_name,
        'prompt': prompt[:15] + '...' if len(prompt) > 15 else prompt,  # Truncate long prompts for display
        'n_tokens': num_tokens,
        'temperature': temperature,
        'top_k': top_k,
        'top_p': top_p,
        'seed': seed
    }
    
    # Generate sequences multiple times
    for generation_round in range(1, num_generations + 1):
        logger.info(f"Generation round {generation_round}/{num_generations}")
        
        results = generate(
            model=inference_wrapped_model,
            max_batch_size=1,  # vortex only supports batch size 1
            tokenizer=mcore_tokenizer,
            prompts=[prompt],
            random_seed=seed,
            inference_params=CommonInferenceParams(
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                return_log_probs=False,
                num_tokens_to_generate=num_tokens,
            ),
        )
        
        # Debug: Log the structure of results
        logger.debug(f"Results type: {type(results)}")
        if results:
            logger.debug(f"First result type: {type(results[0])}")
            if hasattr(results[0], '__dict__'):
                logger.debug(f"First result attributes: {results[0].__dict__.keys()}")
        
        # Extract generated sequences from results
        generated_sequences = []
        for result in results:
            # Based on the output format, we need to extract the generated_text attribute
            if hasattr(result, 'generated_text'):
                # The generated_text contains only the generated part (not the prompt)
                generated_sequences.append(result.generated_text)
            elif isinstance(result, dict) and 'generated_text' in result:
                generated_sequences.append(result['generated_text'])
            else:
                # Fallback to the full text if structure is different
                logger.warning("Could not find generated_text attribute, using full result")
                generated_sequences.append(str(result))
        
        # Save sequences to CSV
        save_sequences_csv(
            sequences=generated_sequences,
            output_file=output_file,
            hyperparameters=hyperparameters,
            generation_round=generation_round
        )
        
        logger.info(f"Completed generation round {generation_round}")
    
    logger.info(f"All generations complete. Results saved to {output_file}")


def main():
    """Main function for Evo2 inference."""
    # Parse args.
    args = parse_args()
    predict(
        ckpt_name=args.ckpt_name,
        num_tokens=args.num_tokens,
        prompt=args.prompt,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        output_file=args.output_file,
        seed=args.seed if args.seed is not None else random.randint(0, 1000000),
        num_generations=args.num_generations
    )


if __name__ == "__main__":
    main()