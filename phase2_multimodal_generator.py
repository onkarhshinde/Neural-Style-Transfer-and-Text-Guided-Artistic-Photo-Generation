"""
Phase 2: Multimodal Text-Guided Artistic Photo Generation
Combines text prompts with image inputs using CLIP + Diffusion models
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import torch

from clip_text_encoder import CLIPTextEncoder
from diffusion_text_to_image import StableDiffusionGenerator


class MultimodalArtisticGenerator:
    """
    Main orchestrator for Phase 2 multimodal generation
    Combines CLIP text encoding with Stable Diffusion for artistic image generation
    """
    
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize multimodal components
        
        Args:
            device: Device to run models on (cuda or cpu)
        """
        self.device = device
        
        print("\n" + "="*70)
        print("PHASE 2: Multimodal Text-Guided Artistic Photo Generation")
        print("="*70)
        
        # Initialize components
        print("\n[1/2] Initializing CLIP Text Encoder...")
        try:
            self.text_encoder = CLIPTextEncoder(device=device)
        except Exception as e:
            print(f"[ERROR] Failed to load CLIP: {e}")
            sys.exit(1)
        
        print("\n[2/2] Initializing Stable Diffusion Generator...")
        try:
            self.generator = StableDiffusionGenerator(device=device)
        except Exception as e:
            print(f"[ERROR] Failed to load Stable Diffusion: {e}")
            sys.exit(1)
        
        print("\n[OK] All components initialized successfully!")
    
    def preprocess_image(self, image_path, max_size=768):
        """
        Preprocess input photograph
        
        Args:
            image_path: Path to input image
            max_size: Maximum dimension size
            
        Returns:
            PIL.Image: Preprocessed image
        """
        try:
            image = Image.open(image_path).convert("RGB")
            
            # Resize maintaining aspect ratio
            image.thumbnail((max_size, max_size), Image.LANCZOS)
            
            print(f"[OK] Image preprocessed: {image.size}")
            return image
        except Exception as e:
            print(f"[ERROR] Error preprocessing image: {e}")
            raise
    
    def analyze_image_and_prompt(self, image_path, text_prompt):
        """
        Analyze relationship between image and text prompt
        Provides insight into multimodal alignment
        
        Args:
            image_path: Path to input image
            text_prompt: Text description of desired style
            
        Returns:
            dict: Analysis results
        """
        try:
            print(f"\n[*] Analyzing multimodal alignment...")
            
            # Encode image and text
            image_embedding = self.text_encoder.encode_image(image_path)
            text_embedding = self.text_encoder.encode_text(text_prompt)
            
            # Compute similarity
            similarity = self.text_encoder.compute_similarity(
                text_embedding[0], 
                image_embedding
            )
            
            analysis = {
                'image_embedding_shape': image_embedding.shape,
                'text_embedding_shape': text_embedding.shape,
                'image_text_similarity': similarity,
                'text_prompt': text_prompt
            }
            
            print(f"[OK] Image-Text Similarity Score: {similarity:.4f}")
            return analysis
        
        except Exception as e:
            print(f"[ERROR] Error in multimodal analysis: {e}")
            return None
    
    def generate_artistic_image(self,
                               content_image_path,
                               text_prompt,
                               num_inference_steps=50,
                               strength=0.75,
                               guidance_scale=7.5,
                               output_path=None):
        """
        Generate artistic image from content photo and text prompt
        
        Args:
            content_image_path: Path to input photograph
            text_prompt: Text description of artistic style
            num_inference_steps: Number of diffusion steps
            strength: How much to modify the image (0.3-0.9)
            guidance_scale: Classifier-free guidance scale
            output_path: Path to save generated image
            
        Returns:
            PIL.Image: Generated artistic image
        """
        try:
            print(f"\n[*] Generating artistic image...")
            print(f"[*] Content Image: {content_image_path}")
            print(f"[*] Text Prompt: '{text_prompt}'")
            
            # Analyze multimodal alignment
            analysis = self.analyze_image_and_prompt(content_image_path, text_prompt)
            
            # Generate image using image-to-image with text guidance
            artistic_image = self.generator.generate_from_image_and_text(
                image_path=content_image_path,
                prompt=text_prompt,
                num_inference_steps=num_inference_steps,
                strength=strength,
                guidance_scale=guidance_scale
            )
            
            # Save if output path provided
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                artistic_image.save(output_path)
                print(f"[OK] Image saved to: {output_path}")
            
            return artistic_image, analysis
        
        except Exception as e:
            print(f"[ERROR] Error generating artistic image: {e}")
            raise
    
    def generate_multiple_variations(self,
                                    content_image_path,
                                    text_prompt,
                                    num_variations=3,
                                    num_inference_steps=50,
                                    strength=0.75,
                                    guidance_scale=7.5,
                                    output_dir="results"):
        """
        Generate multiple artistic variations with varied settings
        
        Args:
            content_image_path: Path to input photograph
            text_prompt: Text description
            num_variations: Number of variations to generate
            num_inference_steps: Diffusion steps
            strength: Modification strength (will be varied)
            guidance_scale: Guidance scale (will be varied)
            output_dir: Output directory
            
        Returns:
            list: List of generated images
        """
        os.makedirs(output_dir, exist_ok=True)
        images = []
        
        print(f"\n[*] Generating {num_variations} artistic variations with varied settings...")
        
        # Create balanced variation ranges - preserve content but show style differences
        # Strength: moderate range to show artistic variation while preserving content
        strength_min = max(0.3, strength - 0.12)
        strength_max = min(0.95, strength + 0.12)
        strength_variations = np.linspace(strength_min, strength_max, num_variations)
        
        # Guidance: moderate range for different style adherence levels
        guidance_min = max(1.0, guidance_scale - 1.0)
        guidance_max = min(20.0, guidance_scale + 1.0)
        guidance_variations = np.linspace(guidance_min, guidance_max, num_variations)
        
        for i in range(num_variations):
            print(f"\n[*] Generating Variation {i+1}/{num_variations}")
            
            # Use varied parameters for each variation
            var_strength = float(strength_variations[i])
            var_guidance = float(guidance_variations[i])
            
            print(f"[*] Strength: {var_strength:.2f} (base: {strength}), Guidance: {var_guidance:.1f} (base: {guidance_scale})")
            
            output_path = os.path.join(
                output_dir,
                f"variation_{i+1:03d}.png"
            )
            
            # Set different seeds for variation
            torch.manual_seed(i)
            np.random.seed(i)
            
            artistic_image, _ = self.generate_artistic_image(
                content_image_path=content_image_path,
                text_prompt=text_prompt,
                num_inference_steps=num_inference_steps,
                strength=var_strength,
                guidance_scale=var_guidance,
                output_path=output_path
            )
            
            # Print image path for web UI to display
            print(f"[IMAGE_READY] {output_path}")
            
            images.append(artistic_image)
        
        print(f"\n[OK] All variations generated successfully!")
        return images


def main():
    """Main function for Phase 2 CLI"""
    parser = argparse.ArgumentParser(
        description='Phase 2: Text-Guided Artistic Photo Generation'
    )
    
    parser.add_argument('content_image', type=str,
                       help='Path to content/input photograph')
    
    parser.add_argument('text_prompt', type=str,
                       help='Text prompt describing artistic style')
    
    parser.add_argument('output_prefix', type=str,
                       help='Output directory/prefix for generated images')
    
    parser.add_argument('--num_variations', type=int, default=3,
                       help='Number of artistic variations to generate (default: 3)')
    
    parser.add_argument('--num_steps', type=int, default=50,
                       help='Number of diffusion steps (default: 50)')
    
    parser.add_argument('--strength', type=float, default=0.75,
                       help='Modification strength 0.3-0.9 (default: 0.75)')
    
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                       help='Classifier-free guidance scale (default: 7.5)')
    
    parser.add_argument('--device', type=str, default=None,
                       help="Device: 'cuda' or 'cpu' (default: auto-detect)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.content_image):
        print(f"[ERROR] Error: Content image not found: {args.content_image}")
        sys.exit(1)
    
    if not args.text_prompt.strip():
        print("[ERROR] Error: Text prompt cannot be empty")
        sys.exit(1)
    
    # Determine device
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"[*] Using device: {device}")
    
    # Initialize generator
    generator = MultimodalArtisticGenerator(device=device)
    
    # Generate variations
    try:
        images = generator.generate_multiple_variations(
            content_image_path=args.content_image,
            text_prompt=args.text_prompt,
            num_variations=args.num_variations,
            num_inference_steps=args.num_steps,
            strength=args.strength,
            guidance_scale=args.guidance_scale,
            output_dir=args.output_prefix
        )
        
        print(f"\n" + "="*70)
        print(f"[OK] Phase 2 Generation Complete!")
        print(f"[*] Generated {len(images)} artistic variation(s)")
        print(f"[*] Output directory: {args.output_prefix}")
        print("="*70)
        
    except KeyboardInterrupt:
        print("\n\n[CANCELLED] Generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Generation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
