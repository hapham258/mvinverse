import torch
import argparse
import os
import os.path as osp
import numpy as np
from mvinverse.models.mvinverse import MVInverse
from PIL import Image
from torchvision import transforms

def load_images(path):
    sources = []
    if osp.isdir(path):
        print(f"Loading images from directory: {path}")
        filenames = sorted([x for x in os.listdir(path) if x.lower().endswith(('.png', '.jpg', '.jpeg', "JPG", "PNG"))])
        for filename in filenames:
            img_path = osp.join(path, filename)
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                width, height = img.size
                
                # Limit the longest side to 1024
                if max(width, height) > 1024:
                    scaling_factor = 1024 / max(width, height)
                    new_width, new_height = int(width * scaling_factor), int(height * scaling_factor)
                else:
                    new_width, new_height = width, height
                
                # Adjust size to be divisible by 14
                new_w, new_h = new_width // 14 * 14, new_height // 14 * 14
                img = img.resize((new_w, new_h))
                sources.append(img)
            
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    for i, img in enumerate(sources):
        sources[i] = transform(img)
    
    return torch.stack(sources, dim=0)

def load_model(ckpt_path_or_repo_id, device):
    """
    Loads the model from a local checkpoint path or a Hugging Face Hub repository ID.
    """
    if os.path.exists(ckpt_path_or_repo_id):
        print(f"Loading model from local checkpoint: {ckpt_path_or_repo_id}")
        model = MVInverse().to(device).eval()
        weight = torch.load(ckpt_path_or_repo_id, map_location=device, weights_only=False)
        weight = weight["model"] if "model" in weight else weight
        missing, unused = model.load_state_dict(weight, strict=False)
        if len(missing) != 0:
            print(f"Warning: Missing keys found: {missing}")
        if len(unused) != 0:
            print(f"Got unexpected keys: {unused}")
    else:
        print(f"Checkpoint not found locally. Attempting to download from Hugging Face Hub: {ckpt_path_or_repo_id}")
        try:
            model = MVInverse.from_pretrained(ckpt_path_or_repo_id)
            model = model.to(device).eval()
        except Exception as e:
            print(f"Error loading model from Hugging Face Hub: {e}")
            print("Please ensure the repository ID is correct and you have an internet connection.")
            raise
            
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference with the mvinverse model.")
    
    parser.add_argument("--data_path", type=str, default="examples/Courtroom",
                        help="Path to the input image directory or a video file.")
    parser.add_argument("--ckpt", type=str, default="maddog241/mvinverse",
                        help="Path to a local model checkpoint OR a Hugging Face Hub repository ID. ")
    parser.add_argument("--save_path", type=str, default='outputs',
                        help="Directory to save the outputs.")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to run inference on ('cuda' or 'cpu'). Default: 'cuda'")
    parser.add_argument("--num_frames", type=int, default=-1,
                        help="Number of frames to run inference")
                        
    args = parser.parse_args()

    # Prepare model
    print(f"Loading model...")
    device = torch.device(args.device)
    model = load_model(args.ckpt, device)

    # Load images
    imgs = load_images(args.data_path).to(device)
    if imgs is None:
        raise RuntimeError("No image loaded")
    else:
        if args.num_frames > 0:
            imgs = imgs[:args.num_frames]
        print(f"Loaded {len(imgs)} images, size is {imgs.shape[-2:]}")
    albedo_list = []

    # Run inference in chunks
    print("Running model inference...")
    chunk_size = 8
    with torch.no_grad():
        for start in range(0, len(imgs), chunk_size):
            inference_imgs = imgs[start:start + chunk_size]
            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
                res = model(inference_imgs[None])
            albedo_list.append(res["albedo"][0].cpu())
            del res
            print(f"Processed {min(start + chunk_size, len(imgs))}/{len(imgs)} frames", flush=True)
    albedo = torch.cat(albedo_list, dim=0)

    # Save results
    save_dir = args.save_path
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results saved to {save_dir}")
    filenames = sorted([x for x in os.listdir(args.data_path) if x.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if args.num_frames > 0:
        filenames = filenames[:args.num_frames]
    for idx, albedo_frame in enumerate(albedo):
        albedo_frame = Image.fromarray((albedo_frame.cpu().numpy() * 255).clip(0, 255).astype(np.uint8))
        albedo_frame.save(osp.join(save_dir, filenames[idx]))
