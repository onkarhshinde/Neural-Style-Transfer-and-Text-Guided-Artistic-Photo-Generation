"""
CLIP-based Text Encoder for Phase 2
Encodes natural language prompts into semantic embeddings
Uses open_clip as drop-in replacement for OpenAI CLIP
"""

import torch
import numpy as np
from PIL import Image
import open_clip


class CLIPTextEncoder:
    """
    Encodes text prompts and images into shared semantic space using CLIP
    Uses open_clip as a pip-friendly alternative to OpenAI's CLIP
    """
    
    def __init__(self, model_name="ViT-B-32", pretrained="openai", device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize CLIP model and processor
        
        Args:
            model_name: Model variant (ViT-B-32, ViT-B-16, ViT-L-14, etc.)
            pretrained: Pretrained weights (openai, laion400M_e32, etc.)
            device: Device to run model on (cuda or cpu)
        """
        self.device = device
        self.model_name = model_name
        
        try:
            # Load model and preprocessing function from open_clip
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
                device=device
            )
            
            # Get tokenizer
            self.tokenizer = open_clip.get_tokenizer(model_name)
            
            self.model.eval()
            print(f"[OK] CLIP model '{model_name}' loaded successfully (pretrained: {pretrained})")
        except Exception as e:
            print(f"[ERROR] Error loading CLIP model: {e}")
            raise
    
    def encode_text(self, text_prompts):
        """
        Encode text prompts into embeddings
        
        Args:
            text_prompts: String or list of strings describing artistic style
            
        Returns:
            np.ndarray: Text embeddings (batch_size, embedding_dim)
        """
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]
        
        try:
            with torch.no_grad():
                # Tokenize and encode text
                text_tokens = self.tokenizer(text_prompts).to(self.device)
                text_embeddings = self.model.encode_text(text_tokens)
                
                # Normalize embeddings
                text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
                
            return text_embeddings.cpu().numpy()
        except Exception as e:
            print(f"[ERROR] Error encoding text: {e}")
            raise
    
    def encode_image(self, image_path):
        """
        Encode image into embedding
        
        Args:
            image_path: Path to image file
            
        Returns:
            np.ndarray: Image embedding (embedding_dim,)
        """
        try:
            image = Image.open(image_path).convert("RGB")
            with torch.no_grad():
                image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
                image_embedding = self.model.encode_image(image_tensor)
                
                # Normalize embeddings
                image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
            
            return image_embedding.cpu().numpy()[0]
        except Exception as e:
            print(f"[ERROR] Error encoding image: {e}")
            raise
    
    def compute_similarity(self, text_embedding, image_embedding):
        """
        Compute cosine similarity between text and image embeddings
        
        Args:
            text_embedding: Text embedding (embedding_dim,)
            image_embedding: Image embedding (embedding_dim,)
            
        Returns:
            float: Similarity score (0 to 1)
        """
        # Ensure embeddings are numpy arrays
        if isinstance(text_embedding, torch.Tensor):
            text_embedding = text_embedding.cpu().numpy()
        if isinstance(image_embedding, torch.Tensor):
            image_embedding = image_embedding.cpu().numpy()
        
        # If batch, take first element
        if text_embedding.ndim > 1:
            text_embedding = text_embedding[0]
        
        similarity = np.dot(text_embedding, image_embedding)
        return float(similarity)


def test_clip_encoder():
    """Test function for CLIP encoder"""
    print("\n" + "="*60)
    print("Testing CLIP Text Encoder (open_clip)")
    print("="*60)
    
    encoder = CLIPTextEncoder()
    
    # Test text encoding
    prompts = [
        "Van Gogh painting style with swirling brushstrokes",
        "Watercolor painting style",
        "Cinematic photography with dramatic lighting"
    ]
    
    print("\n[1] Encoding text prompts...")
    text_embeddings = encoder.encode_text(prompts)
    print(f"[OK] Text embeddings shape: {text_embeddings.shape}")
    
    print("\n[2] Testing similarity computation...")
    sim = np.dot(text_embeddings[0], text_embeddings[1])
    print(f"[OK] Similarity between first two prompts: {sim:.4f}")
    
    print("\n[OK] CLIP encoder test passed!")

if __name__ == "__main__":
    test_clip_encoder()
