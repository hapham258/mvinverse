import torch
import argparse
import os
import os.path as osp
import numpy as np
from mvinverse.models.mvinverse import MVInverse
from PIL import Image
from torchvision import transforms

def load_images(path):
    if not osp.isdir(path):
        raise RuntimeError(f"Not a directory: {path}")
    print(f"Loading images from directory: {path}")
    filenames = sorted(x for x in os.listdir(path) if x.lower().endswith(('.png', '.jpg', '.jpeg', '.ppm')))
    if not filenames:
        raise RuntimeError(f"No images found in {path}")
    return filenames

def preprocess_image(img_path):
    with Image.open(img_path) as img:
        img = img.convert('RGB')
        width, height = img.size

        # Limit the longest side to 1024
        if max(width, height) > 1024:
            scaling_factor = 1024 / max(width, height)
            width, height = int(width * scaling_factor), int(height * scaling_factor)

        # Adjust size to be divisible by 14
        width, height = width // 14 * 14, height // 14 * 14
        img = img.resize((width, height))
        return transforms.ToTensor()(img)

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
                        help="Path to a local model checkpoint OR a Hugging Face Hub repository ID.")
    parser.add_argument("--save_path", type=str, default='outputs',
                        help="Directory to save the outputs.")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to run inference on ('cuda' or 'cpu'). Default: 'cuda'")
    parser.add_argument("--num_frames", type=int, default=-1,
                        help="Number of frames to run inference")
    parser.add_argument("--chunk_size", type=int, default=16,
                        help="Number of images to process per inference batch.")
    args = parser.parse_args()

    # Prepare model
    print("Loading model...")
    device = torch.device(args.device)
    model = load_model(args.ckpt, device)

    # Load image filenames
    filenames = load_images(args.data_path)
    if args.num_frames > 0:
        filenames = filenames[:args.num_frames]
    print(f"Processing {len(filenames)} images")

    # Prepare output directory
    save_dir = args.save_path
    os.makedirs(save_dir, exist_ok=True)
    print(f"Results saved to {save_dir}")

    # Run inference in chunks
    print("Running model inference...")
    chunk_size = args.chunk_size
    transform = transforms.ToTensor()
    with torch.no_grad():
        for start in range(0, len(filenames), chunk_size):
            batch_names = filenames[start:start + chunk_size]
            imgs = torch.stack([
                preprocess_image(osp.join(args.data_path, name))
                for name in batch_names
            ]).to(device)

            with torch.amp.autocast('cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
                res = model(imgs[None])

            albedo = res["albedo"][0].cpu()
            del res, imgs

            for filename, albedo_frame in zip(batch_names, albedo):
                albedo_frame = Image.fromarray(
                    (albedo_frame.numpy() * 255).clip(0, 255).astype(np.uint8)
                )
                output_name = osp.splitext(filename)[0] + '.png'
                albedo_frame.save(osp.join(save_dir, output_name))

            if (start // chunk_size + 1) % 10 == 0 or start + chunk_size >= len(filenames):
                print(f"Processed {min(start + chunk_size, len(filenames))}/{len(filenames)} frames", flush=True)