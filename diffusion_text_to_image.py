"""
Stable Diffusion Model Wrapper for Phase 2
Text-guided image generation using Stable Diffusion
"""

import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import os


class StableDiffusionGenerator:
    """
    Generates artistic images from text prompts using Stable Diffusion
    """
    
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize Stable Diffusion pipeline
        
        Args:
            model_id: HuggingFace model ID for Stable Diffusion
            device: Device to run model on (cuda or cpu)
        """
        self.device = device
        self.model_id = model_id
        
        print(f"[*] Loading Stable Diffusion model: {model_id}")
        print(f"[*] Device: {device}")
        
        try:
            # Load pipeline in half precision to save memory
            dtype = torch.float16 if device == "cuda" else torch.float32
            
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                safety_checker=None  # Disable safety checker for faster inference
            )
            
            # Use faster scheduler for faster generation
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config
            )
            
            self.pipe = self.pipe.to(device)
            
            # Memory optimization - reduce to enable_attention_slicing for GPU too
            self.pipe.enable_attention_slicing()
            
            # Additional GPU memory optimization
            if device == "cuda":
                try:
                    self.pipe.enable_memory_efficient_attention()
                except:
                    pass  # Not available in all versions
            
            print("[OK] Stable Diffusion model loaded successfully")
            
        except Exception as e:
            print(f"[ERROR] Error loading Stable Diffusion model: {e}")
            raise
    
    def generate_from_text(self, 
                          prompt, 
                          num_inference_steps=50,
                          strength=0.8,
                          guidance_scale=7.5,
                          height=512,
                          width=512,
                          seed=None):
        """
        Generate image from text prompt
        
        Args:
            prompt: Text description of desired image
            num_inference_steps: Number of diffusion steps (more = better quality, slower)
            strength: How much to modify the image (0.0 = no change, 1.0 = full change)
            guidance_scale: Scale for classifier-free guidance (higher = more adherence to prompt)
            height: Output image height (multiple of 8)
            width: Output image width (multiple of 8)
            seed: Random seed for reproducibility
            
        Returns:
            PIL.Image: Generated image
        """
        try:
            # Set seed for reproducibility
            if seed is not None:
                torch.manual_seed(seed)
                np.random.seed(seed)
            
            print(f"\n[*] Generating image from prompt: '{prompt}'")
            print(f"[*] Steps: {num_inference_steps}, Guidance: {guidance_scale}")
            
            with torch.no_grad():
                output = self.pipe(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width
                )
            
            image = output.images[0]
            print("[OK] Image generation completed successfully")
            
            return image
        
        except Exception as e:
            print(f"[ERROR] Error generating image: {e}")
            raise
    
    def generate_from_image_and_text(self,
                                    image_path,
                                    prompt,
                                    num_inference_steps=50,
                                    strength=0.75,
                                    guidance_scale=7.5,
                                    seed=None):
        """
        Generate image using image-to-image with text guidance (img2img)
        
        This combines the structure of the input image with the artistic style
        described in the text prompt.
        
        Args:
            image_path: Path to input photograph
            prompt: Text description of artistic style to apply
            num_inference_steps: Number of diffusion steps
            strength: How much to modify the image (0.3-0.9 recommended)
            guidance_scale: Scale for classifier-free guidance
            seed: Random seed
            
        Returns:
            PIL.Image: Stylized image
        """
        try:
            from diffusers import StableDiffusionImg2ImgPipeline
            
            print(f"\n[*] Loading Image-to-Image pipeline...")
            
            # Use float32 for stable generation on both CPU and GPU
            dtype = torch.float32
            
            # Use img2img pipeline
            pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                safety_checker=None
            )
            pipe_img2img.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe_img2img.scheduler.config
            )
            pipe_img2img = pipe_img2img.to(self.device)
            
            # Memory optimization
            pipe_img2img.enable_attention_slicing()
            
            if self.device == "cuda":
                try:
                    pipe_img2img.enable_memory_efficient_attention()
                except:
                    pass
            
            # Load and prepare image
            init_image = Image.open(image_path).convert("RGB")
            
            # Resize to standard size (512x512) for better results
            init_image = init_image.resize((512, 512), Image.LANCZOS)
            
            print(f"[*] Input image size: {init_image.size}")
            
            # Set seed for reproducibility
            if seed is not None:
                torch.manual_seed(seed)
                np.random.seed(seed)
                if self.device == "cuda":
                    torch.cuda.manual_seed(seed)
            
            print(f"[*] Applying text-guided style: '{prompt}'")
            print(f"[*] Strength: {strength}, Guidance: {guidance_scale}, Steps: {num_inference_steps}")
            
            # Ensure strength is in valid range
            strength = max(0.1, min(1.0, strength))
            
            with torch.no_grad():
                output = pipe_img2img(
                    prompt=prompt,
                    image=init_image,
                    num_inference_steps=num_inference_steps,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    generator=torch.Generator(device=self.device).manual_seed(seed if seed else 0)
                )
            
            image = output.images[0]
            print("[OK] Image-to-image generation completed successfully")
            
            return image
        
        except Exception as e:
            print(f"[ERROR] Error in image-to-image generation: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def generate_batch(self,
                      prompts,
                      num_inference_steps=50,
                      guidance_scale=7.5,
                      height=512,
                      width=512):
        """
        Generate multiple images from different prompts
        
        Args:
            prompts: List of text prompts
            num_inference_steps: Number of diffusion steps
            guidance_scale: Guidance scale
            height: Output height
            width: Output width
            
        Returns:
            list: List of generated PIL Images
        """
        images = []
        for i, prompt in enumerate(prompts, 1):
            print(f"\n[*] Generating image {i}/{len(prompts)}...")
            img = self.generate_from_text(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width
            )
            images.append(img)
        
        return images


def test_diffusion_generator():
    """Test function for Stable Diffusion generator"""
    print("\n" + "="*60)
    print("Testing Stable Diffusion Generator")
    print("="*60)
    
    generator = StableDiffusionGenerator()
    
    # Test text-to-image generation
    test_prompt = "A beautiful sunset over mountains, oil painting style"
    print(f"\n[*] Test prompt: '{test_prompt}'")
    
    image = generator.generate_from_text(
        prompt=test_prompt,
        num_inference_steps=10,  # Low for testing
        guidance_scale=7.5,
        height=512,
        width=512
    )
    
    # Save test image
    os.makedirs("test_outputs", exist_ok=True)
    image.save("test_outputs/test_generation.png")
    print("[OK] Test image saved to test_outputs/test_generation.png")


if __name__ == "__main__":
    test_diffusion_generator()
