#!/usr/bin/env python3
"""
Debug script to reproduce and diagnose NCCL timeout during evaluation.
This script simulates what happens during validation to find the root cause.

Usage:
    # Single GPU test (baseline)
    python scripts/debug_eval_timeout.py --mode single
    
    # Multi-GPU test (reproduce the issue)
    torchrun --nproc_per_node=2 scripts/debug_eval_timeout.py --mode multi
    
    # Test with actual validation data
    torchrun --nproc_per_node=2 scripts/debug_eval_timeout.py --mode eval_data
"""

import os
import sys
import time
import argparse
import json
import torch
import torch.distributed as dist
from PIL import Image

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def print_rank0(msg, rank=0):
    """Print only on rank 0."""
    if rank == 0:
        print(msg)


def test_basic_communication(rank, world_size):
    """Test 1: Basic NCCL communication."""
    print(f"[Rank {rank}] Test 1: Basic AllReduce communication")
    
    tensor = torch.ones(1000, 1000, device='cuda') * (rank + 1)
    
    start = time.time()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    expected = sum(range(1, world_size + 1))
    actual = tensor[0, 0].item()
    
    print(f"[Rank {rank}] AllReduce completed in {elapsed:.4f}s, "
          f"expected={expected}, actual={actual}, match={abs(expected-actual)<0.01}")
    
    dist.barrier()
    print(f"[Rank {rank}] Test 1 PASSED")


def test_uneven_workload(rank, world_size):
    """Test 2: Simulate uneven workload (different processing times per rank)."""
    print(f"[Rank {rank}] Test 2: Uneven workload simulation")
    
    # Rank 1 takes longer (simulating larger images)
    sleep_time = 2.0 if rank == 1 else 0.5
    print(f"[Rank {rank}] Sleeping for {sleep_time}s to simulate workload...")
    time.sleep(sleep_time)
    
    # Now try to synchronize
    tensor = torch.ones(1000, device='cuda') * rank
    
    start = time.time()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"[Rank {rank}] AllReduce after uneven workload: {elapsed:.4f}s")
    
    dist.barrier()
    print(f"[Rank {rank}] Test 2 PASSED")


def test_uneven_batch_count(rank, world_size):
    """Test 3: Simulate uneven batch counts (the most likely cause)."""
    print(f"[Rank {rank}] Test 3: Uneven batch count simulation")
    
    # Rank 0 has 10 batches, Rank 1 has 9 batches
    num_batches = 10 if rank == 0 else 9
    
    print(f"[Rank {rank}] Processing {num_batches} batches...")
    
    for i in range(num_batches):
        # Simulate batch processing
        tensor = torch.randn(100, 100, device='cuda')
        tensor = tensor @ tensor.T  # Some computation
        
        # This is where the hang would occur - rank 0 tries AllReduce on batch 10
        # but rank 1 has already exited the loop
        print(f"[Rank {rank}] Batch {i+1}/{num_batches} done")
    
    # Barrier to sync - this should work if both ranks reach here
    print(f"[Rank {rank}] Reaching barrier after loop...")
    dist.barrier()
    print(f"[Rank {rank}] Test 3 PASSED (no hang)")


def test_checkpoint_save(rank, world_size):
    """Test 4: Simulate checkpoint saving (only rank 0 saves, others wait)."""
    print(f"[Rank {rank}] Test 4: Checkpoint save simulation")
    
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from peft import LoraConfig, get_peft_model
    import tempfile
    import shutil
    
    model_path = "/scratch/youliang/qwen2.5-vl-7b"
    
    print(f"[Rank {rank}] Loading model with LoRA...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).cuda()
    
    # Add LoRA (similar to training setup)
    lora_config = LoraConfig(
        r=64,
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
    )
    model = get_peft_model(model, lora_config)
    
    processor = AutoProcessor.from_pretrained(model_path)
    
    print(f"[Rank {rank}] Model with LoRA loaded")
    
    # Sync before save test
    dist.barrier()
    
    # Create temp directory for saving
    save_dir = tempfile.mkdtemp(prefix=f"checkpoint_test_rank{rank}_")
    
    print(f"[Rank {rank}] Starting checkpoint save test...")
    print(f"[Rank {rank}] Save directory: {save_dir}")
    
    save_start = time.time()
    
    # Only rank 0 saves (this is the typical pattern that causes timeout)
    if rank == 0:
        print(f"[Rank 0] Saving model checkpoint...")
        
        # Save LoRA weights
        model.save_pretrained(save_dir)
        processor.save_pretrained(save_dir)
        
        # Simulate additional save operations (optimizer states, etc.)
        # In real training, this could include DeepSpeed optimizer states
        dummy_state = {f"layer_{i}": torch.randn(1000, 1000) for i in range(100)}
        torch.save(dummy_state, os.path.join(save_dir, "optimizer_state.bin"))
        
        save_elapsed = time.time() - save_start
        print(f"[Rank 0] Checkpoint saved in {save_elapsed:.2f}s")
    else:
        print(f"[Rank {rank}] Waiting for rank 0 to save...")
    
    # This is where the timeout could occur!
    # Rank 1 reaches barrier quickly, rank 0 might still be saving
    print(f"[Rank {rank}] Reaching barrier after save...")
    barrier_start = time.time()
    
    dist.barrier()
    
    barrier_elapsed = time.time() - barrier_start
    total_elapsed = time.time() - save_start
    
    print(f"[Rank {rank}] Barrier wait time: {barrier_elapsed:.2f}s")
    print(f"[Rank {rank}] Total save phase time: {total_elapsed:.2f}s")
    
    # Cleanup
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    
    dist.barrier()
    print(f"[Rank {rank}] Test 4 PASSED")
    
    del model
    torch.cuda.empty_cache()


def test_deepspeed_checkpoint_save(rank, world_size):
    """Test 5: Simulate DeepSpeed-style checkpoint saving with ZeRO."""
    print(f"[Rank {rank}] Test 5: DeepSpeed checkpoint save simulation")
    
    import tempfile
    import shutil
    
    # Create temp directory
    save_dir = tempfile.mkdtemp(prefix=f"ds_checkpoint_test_")
    
    print(f"[Rank {rank}] Simulating DeepSpeed ZeRO-2 checkpoint save...")
    
    # Simulate optimizer states (ZeRO-2 partitions these across ranks)
    # Each rank saves its own partition
    optimizer_state_size_mb = 500  # Simulate 500MB per rank
    
    save_start = time.time()
    
    # Each rank saves its partition
    my_partition = torch.randn(optimizer_state_size_mb * 1024 * 1024 // 4, device='cpu')  # ~500MB
    partition_path = os.path.join(save_dir, f"optimizer_state_rank{rank}.bin")
    
    print(f"[Rank {rank}] Saving optimizer partition ({optimizer_state_size_mb}MB)...")
    torch.save(my_partition, partition_path)
    
    partition_save_time = time.time() - save_start
    print(f"[Rank {rank}] Partition saved in {partition_save_time:.2f}s")
    
    # Barrier after partition save
    dist.barrier()
    print(f"[Rank {rank}] All partitions saved")
    
    # Now only rank 0 consolidates and saves final checkpoint
    if rank == 0:
        print(f"[Rank 0] Consolidating and saving final model...")
        
        # Simulate model save (this is the slow part)
        model_state_size_mb = 1000  # 1GB model state
        model_state = torch.randn(model_state_size_mb * 1024 * 1024 // 4, device='cpu')
        model_path = os.path.join(save_dir, "model_state.bin")
        
        torch.save(model_state, model_path)
        
        consolidate_time = time.time() - save_start - partition_save_time
        print(f"[Rank 0] Model consolidated in {consolidate_time:.2f}s")
    else:
        print(f"[Rank {rank}] Waiting for rank 0 to consolidate...")
    
    # Critical barrier - this is where timeout occurs!
    print(f"[Rank {rank}] Reaching final barrier...")
    barrier_start = time.time()
    
    dist.barrier()
    
    barrier_time = time.time() - barrier_start
    total_time = time.time() - save_start
    
    print(f"[Rank {rank}] Final barrier wait: {barrier_time:.2f}s")
    print(f"[Rank {rank}] Total checkpoint time: {total_time:.2f}s")
    
    # Check if this would have caused timeout (600s default)
    if barrier_time > 60:
        print(f"[Rank {rank}] ⚠️  WARNING: Barrier wait > 60s, could timeout with slow disk!")
    
    # Cleanup
    dist.barrier()
    if rank == 0 and os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    
    print(f"[Rank {rank}] Test 5 PASSED")


def test_model_forward(rank, world_size):
    """Test 4: Load actual model and run forward pass."""
    print(f"[Rank {rank}] Test 4: Model forward pass test")
    
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    
    model_path = "/scratch/youliang/qwen2.5-vl-7b"
    
    print(f"[Rank {rank}] Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).cuda()
    model.eval()
    
    processor = AutoProcessor.from_pretrained(model_path)
    
    print(f"[Rank {rank}] Model loaded successfully")
    
    # Create dummy input
    dummy_text = "What is shown in this image?"
    
    # Create a dummy image (different sizes per rank to simulate dynamic resolution)
    img_size = (1280, 720) if rank == 0 else (640, 480)
    dummy_image = Image.new('RGB', img_size, color='white')
    
    print(f"[Rank {rank}] Processing image of size {img_size}...")
    
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": dummy_text}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=[text], images=[dummy_image], return_tensors="pt").to('cuda')
    
    print(f"[Rank {rank}] Input tokens: {inputs['input_ids'].shape}")
    
    # Forward pass
    start = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"[Rank {rank}] Forward pass: {elapsed:.4f}s, loss shape: {outputs.loss if hasattr(outputs, 'loss') else 'N/A'}")
    
    # AllReduce to sync
    dummy_loss = torch.tensor([1.0], device='cuda')
    dist.all_reduce(dummy_loss, op=dist.ReduceOp.SUM)
    
    dist.barrier()
    print(f"[Rank {rank}] Test 4 PASSED")
    
    del model
    torch.cuda.empty_cache()


def test_eval_data_distribution(rank, world_size):
    """Test 5: Check actual eval data distribution across ranks."""
    print(f"[Rank {rank}] Test 5: Eval data distribution check")
    
    eval_path = "/home/agent/mobiAgent/MobiAgent/tools_for_reproduction/generated_data/sft_data/mobimind_decider_val.json"
    
    with open(eval_path, 'r') as f:
        data = json.load(f)
    
    total_samples = len(data)
    samples_per_rank = total_samples // world_size
    remainder = total_samples % world_size
    
    # Calculate what each rank would get
    if rank < remainder:
        my_samples = samples_per_rank + 1
        start_idx = rank * (samples_per_rank + 1)
    else:
        my_samples = samples_per_rank
        start_idx = remainder * (samples_per_rank + 1) + (rank - remainder) * samples_per_rank
    
    print(f"[Rank {rank}] Total samples: {total_samples}")
    print(f"[Rank {rank}] Samples per rank (base): {samples_per_rank}")
    print(f"[Rank {rank}] Remainder: {remainder}")
    print(f"[Rank {rank}] My samples: {my_samples} (indices {start_idx} to {start_idx + my_samples - 1})")
    
    # Check if this could cause issues
    if remainder != 0:
        print(f"[Rank {rank}] ⚠️  WARNING: Uneven distribution detected! "
              f"This can cause NCCL timeout if not handled properly.")
    else:
        print(f"[Rank {rank}] ✓ Even distribution")
    
    # Simulate processing with actual sample counts
    print(f"[Rank {rank}] Simulating {my_samples} sample processing...")
    for i in range(my_samples):
        if i % 50 == 0:
            print(f"[Rank {rank}] Sample {i}/{my_samples}")
        time.sleep(0.01)  # Simulate work
    
    print(f"[Rank {rank}] Done processing, reaching barrier...")
    dist.barrier()
    print(f"[Rank {rank}] Test 5 PASSED")


def test_single_gpu():
    """Run tests on single GPU (baseline)."""
    print("=" * 60)
    print("Single GPU Test (Baseline)")
    print("=" * 60)
    
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    
    model_path = "/scratch/youliang/qwen2.5-vl-7b"
    
    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).cuda()
    model.eval()
    
    processor = AutoProcessor.from_pretrained(model_path)
    print("Model loaded successfully")
    
    # Test with different image sizes
    for size in [(640, 480), (1280, 720), (1920, 1080)]:
        dummy_image = Image.new('RGB', size, color='white')
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image."}]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=[text], images=[dummy_image], return_tensors="pt").to('cuda')
        
        start = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        print(f"Image {size}: {elapsed:.4f}s, tokens: {inputs['input_ids'].shape[1]}")
    
    print("Single GPU test completed successfully!")


def main():
    parser = argparse.ArgumentParser(description='Debug NCCL timeout during evaluation')
    parser.add_argument('--mode', choices=['single', 'multi', 'eval_data'], 
                        default='multi', help='Test mode')
    args = parser.parse_args()
    
    if args.mode == 'single':
        test_single_gpu()
        return
    
    # Multi-GPU tests
    rank, world_size, local_rank = setup_distributed()
    
    try:
        print(f"\n{'='*60}")
        print(f"[Rank {rank}] Debug NCCL Timeout - Multi-GPU Tests")
        print(f"[Rank {rank}] World size: {world_size}, Local rank: {local_rank}")
        print(f"{'='*60}\n")
        
        # Run tests
        test_basic_communication(rank, world_size)
        print()
        
        test_uneven_workload(rank, world_size)
        print()
        
        test_uneven_batch_count(rank, world_size)
        print()
        
        if args.mode == 'eval_data':
            test_eval_data_distribution(rank, world_size)
            print()
        
        # New checkpoint save tests (Problem 2)
        test_checkpoint_save(rank, world_size)
        print()
        
        test_deepspeed_checkpoint_save(rank, world_size)
        print()
        
        test_model_forward(rank, world_size)
        print()
        
        print_rank0("\n" + "="*60, rank)
        print_rank0("ALL TESTS PASSED!", rank)
        print_rank0("="*60 + "\n", rank)
        
    except Exception as e:
        print(f"[Rank {rank}] ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
