#!/usr/bin/env python
"""
Phase 2 Multimodal Artistic Photo Generator - CPU-Optimized Version
Optimized for CPU execution without CUDA
Safe for Python 3.7 with PyTorch 1.13.1+cpu
"""

import os
import sys
import torch
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import time
import json

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clip_text_encoder import CLIPTextEncoder
from diffusion_text_to_image import StableDiffusionGenerator


class MultimodalArtisticGeneratorCPU:
    """
    Multimodal generator with CPU optimizations.
    
    Settings optimized for CPU execution:
    - Reduced memory footprint
    - EnableAttentionSlicing for CPU
    - Optimized batch sizes
    """
    
    def __init__(self, device=None, dtype=torch.float32, enable_fp16=False):
        """
        Initialize generator with CPU optimizations.
        
        Args:
            device (str): 'cpu' or 'cuda' (default: auto-detect)
            dtype: torch.float32 (recommended for CPU)
            enable_fp16 (bool): Use half precision (slower on CPU, skip it)
        """
        # Force CPU if CUDA not available
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if device == 'cuda' and not torch.cuda.is_available():
            print("[WARNING] CUDA requested but not available. Using CPU.")
            device = 'cpu'
        
        self.device = device
        self.dtype = dtype if device == 'cpu' else torch.float32
        
        print(f"[*] Initializing Multimodal Generator")
        print(f"    Device: {device}")
        print(f"    Data type: {self.dtype}")
        print()
        
        # Initialize components
        print("[*] Loading CLIP Text Encoder...")
        self.text_encoder = CLIPTextEncoder(device=device)
        print("[OK] CLIP Text Encoder loaded")
        print()
        
        print("[*] Loading Stable Diffusion Generator...")
        self.diffusion_gen = StableDiffusionGenerator(
            device=device,
            enable_memory_efficient_attention=True,
            enable_attention_slicing=True
        )
        print("[OK] Stable Diffusion Generator loaded")
        print()
    
    def generate_artistic_image(
        self,
        content_image_path,
        text_prompt,
        output_prefix,
        num_steps=20,
        strength=0.75,
        guidance_scale=7.5,
        num_variations=1,
        seed=None,
        callback=None
    ):
        """
        Generate artistic image with CPU optimizations.
        
        Args:
            content_image_path (str): Path to input image
            text_prompt (str): Text description
            output_prefix (str): Output path/prefix
            num_steps (int): Inference steps (recommend 20 for CPU)
            strength (float): 0.0-1.0, blend amount
            guidance_scale (float): Text guidance strength
            num_variations (int): Number of variations (recommend 1 for CPU)
            seed (int): Random seed for reproducibility
            callback (function): Progress callback
        
        Returns:
            dict: Generation results
        """
        print(f"\n========== MULTIMODAL GENERATION START ==========")
        print(f"Input Image: {content_image_path}")
        print(f"Prompt: {text_prompt}")
        print(f"Output Dir: {output_prefix}")
        print(f"Steps: {num_steps}")
        print(f"Strength: {strength}")
        print(f"Guidance: {guidance_scale}")
        print(f"Variations: {num_variations}")
        if seed:
            print(f"Seed: {seed}")
        print(f"Device: {self.device}")
        print("=" * 50)
        print()
        
        # Create output directory
        output_dir = Path(output_prefix)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load input image
        print(f"[*] Loading image: {content_image_path}")
        image = Image.open(content_image_path).convert('RGB')
        original_size = image.size
        print(f"    Original size: {original_size}")
        print()
        
        # Resize for processing (CPU-friendly)
        processing_size = (512, 512)  # Standard size
        image_resized = image.resize(processing_size, Image.Resampling.LANCZOS)
        print(f"[*] Resized to: {processing_size}")
        
        # Encode text prompt
        print(f"[*] Encoding text prompt...")
        text_embedding = self.text_encoder.encode_text(text_prompt)
        print(f"    Embedding shape: {text_embedding.shape}")
        print(f"    Embedding dtype: {text_embedding.dtype}")
        print()
        
        # Encode input image
        print(f"[*] Encoding image...")
        image_embedding = self.text_encoder.encode_image(content_image_path)
        print(f"    Image embedding shape: {image_embedding.shape}")
        
        # Compute similarity
        similarity = self.text_encoder.compute_similarity(
            text_embedding, 
            image_embedding
        )
        print(f"    Text-Image Similarity: {similarity:.4f}")
        print()
        
        # Generate variations
        print(f"[*] Starting generation pipeline...")
        print(f"    Generating {num_variations} variation(s)")
        print()
        
        generated_images = []
        metadata_list = []
        
        start_time = time.time()
        
        for var_idx in range(num_variations):
            print(f"----- Variation {var_idx + 1}/{num_variations} -----")
            var_start = time.time()
            
            # Set seed for reproducibility
            if seed is not None:
                var_seed = seed + var_idx
            else:
                var_seed = np.random.randint(0, 2**32 - 1)
            
            torch.manual_seed(var_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(var_seed)
            
            print(f"[*] Seed: {var_seed}")
            print(f"[*] Generating with text guidance...")
            
            try:
                # Generate image with text guidance
                generated = self.diffusion_gen.generate_from_image_and_text(
                    image_resized,
                    text_prompt,
                    num_inference_steps=num_steps,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    seed=var_seed
                )
                
                # Save result
                output_filename = f"{output_prefix}_var{var_idx+1}.png"
                generated.save(output_filename)
                generated_images.append(generated)
                print(f"[OK] Saved: {output_filename}")
                
                var_time = time.time() - var_start
                print(f"    Time: {var_time:.1f}s")
                print()
                
                # Metadata
                metadata_list.append({
                    'variation': var_idx + 1,
                    'seed': var_seed,
                    'filename': output_filename,
                    'steps': num_steps,
                    'strength': strength,
                    'guidance_scale': guidance_scale,
                    'time_seconds': var_time
                })
                
            except Exception as e:
                print(f"[ERROR] Generation failed: {str(e)}")
                print()
                continue
        
        total_time = time.time() - start_time
        
        # Save metadata
        metadata = {
            'prompt': text_prompt,
            'input_image': content_image_path,
            'input_size': original_size,
            'processing_size': processing_size,
            'text_image_similarity': float(similarity),
            'device': self.device,
            'total_time_seconds': total_time,
            'variations': metadata_list
        }
        
        metadata_file = f"{output_prefix}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"[*] Metadata saved: {metadata_file}")
        
        print()
        print("=" * 50)
        print(f"[OK] Generation complete!")
        print(f"    Total time: {total_time:.1f}s")
        print(f"    Avg per variation: {total_time/num_variations:.1f}s")
        print(f"    Generated: {len(generated_images)} images")
        print("=" * 50)
        
        return {
            'success': True,
            'images': generated_images,
            'metadata': metadata,
            'output_dir': str(output_dir)
        }
    
    def generate_multiple_variations(
        self,
        content_image_path,
        text_prompts,
        output_dir,
        num_steps=20,
        num_variations_per_prompt=1,
        **kwargs
    ):
        """
        Generate multiple variations for multiple prompts (CPU-friendly batch).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = []
        
        for prompt_idx, prompt in enumerate(text_prompts):
            print(f"\n{'='*70}")
            print(f"Processing prompt {prompt_idx + 1}/{len(text_prompts)}")
            print(f"Prompt: {prompt}")
            print(f"{'='*70}\n")
            
            output_prefix = str(output_dir / f"prompt_{prompt_idx+1}")
            
            result = self.generate_artistic_image(
                content_image_path=content_image_path,
                text_prompt=prompt,
                output_prefix=output_prefix,
                num_steps=num_steps,
                num_variations=num_variations_per_prompt,
                **kwargs
            )
            
            all_results.append(result)
        
        return all_results


def main():
    """CLI interface for CPU-optimized generator."""
    parser = argparse.ArgumentParser(
        description='Phase 2 Multimodal Artistic Photo Generator (CPU-Optimized)'
    )
    
    parser.add_argument('content_image', help='Path to content image')
    parser.add_argument('text_prompt', help='Text description for styling')
    parser.add_argument('output_prefix', help='Output path/prefix for results')
    
    parser.add_argument('--num_variations', type=int, default=1,
                        help='Number of variations to generate (default: 1 for CPU)')
    parser.add_argument('--num_steps', type=int, default=20,
                        help='Inference steps (default: 20 for CPU)')
    parser.add_argument('--strength', type=float, default=0.75,
                        help='Content preservation strength (default: 0.75)')
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                        help='Text guidance strength (default: 7.5)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default=None,
                        help='Device: cpu or cuda (default: auto)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.content_image):
        print(f"[ERROR] Image not found: {args.content_image}")
        sys.exit(1)
    
    # Initialize generator
    try:
        generator = MultimodalArtisticGeneratorCPU(device=args.device)
    except Exception as e:
        print(f"[ERROR] Failed to initialize generator: {str(e)}")
        sys.exit(1)
    
    # Generate
    try:
        result = generator.generate_artistic_image(
            content_image_path=args.content_image,
            text_prompt=args.text_prompt,
            output_prefix=args.output_prefix,
            num_steps=args.num_steps,
            strength=args.strength,
            guidance_scale=args.guidance_scale,
            num_variations=args.num_variations,
            seed=args.seed
        )
        
        if result['success']:
            print("\n[OK] Generation successful!")
            print(f"Output directory: {result['output_dir']}")
        else:
            print("\n[ERROR] Generation failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n[ERROR] Generation error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
