import collections
import json
import random
import torch
from transformers import CLIPTokenizer, CLIPModel
from typing import List, Tuple, Optional, Dict, Any
import os


from obj_utils import parse_obj_file, calculate_bounding_box

MODEL_INFO_PATH = "examples/datasets/front_3d_with_improved_mat/3D-FUTURE-model"

def _ensure_clip_available() -> None:
    """Ensure required dependencies for CLIP (via Hugging Face) are available."""
    if CLIPTokenizer is None or CLIPModel is None or torch is None:
        raise RuntimeError(
            "Hugging Face Transformers CLIP not available. Install with: pip install torch torchvision torchaudio transformers"
        )


def load_clip_model(
    model_name: str = "openai/clip-vit-base-patch32",
    device: Optional[str] = None,
) -> Tuple[Any, Any, str]:
    """
    Load a CLIP text model/tokenizer for text-text similarity using Hugging Face.

    Returns (model, tokenizer, device_str)
    """
    _ensure_clip_available()
    device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CLIPTokenizer.from_pretrained(model_name)  # type: ignore
    model = CLIPModel.from_pretrained(model_name)  # type: ignore
    model = model.to(device_str)
    model.eval()
    return model, tokenizer, device_str


def compute_text_cosine_similarities(
    query: str,
    candidates: List[str],
    model: Any,
    device: str,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Encode `query` and `candidates` with CLIP text encoder (HF) and compute cosine similarities.

    Returns a 1D tensor of size len(candidates) with cosine similarities in [-1, 1].
    """
    _ensure_clip_available()
    assert isinstance(candidates, list) and len(candidates) > 0, "candidates must be a non-empty list"

    # Expect that caller passed the HF CLIPModel; we can obtain features via get_text_features.
    with torch.no_grad():
        # Tokenize separately to avoid padding query to length of candidates
        tokenizer = None
        # If the caller passed a tuple (model, tokenizer), handle that gracefully
        if isinstance(model, tuple) and len(model) == 2:
            model, tokenizer = model
        # Fallback: try to read tokenizer from global if available
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided alongside model.")

        query_inputs = tokenizer([query], return_tensors="pt", padding=True).to(device)
        cand_inputs = tokenizer(candidates, return_tensors="pt", padding=True).to(device)

        text_query = model.get_text_features(**query_inputs)
        text_cands = model.get_text_features(**cand_inputs)

        if normalize:
            text_query = text_query / text_query.norm(dim=-1, keepdim=True)
            text_cands = text_cands / text_cands.norm(dim=-1, keepdim=True)

        # Cosine similarity of normalized embeddings equals dot product
        sims = (text_query @ text_cands.T).squeeze(0).float().cpu()
        return sims


def select_category_with_clip(
    word: str,
    categories: List[str],
    threshold: float,
    top_k: int = 5,
    model_name: str = "openai/clip-vit-base-patch32",
    device: Optional[str] = None,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    """
    Given a `word` and list of `categories`, compute CLIP text similarities and:
    - Take the top-K categories whose similarity >= threshold.
    - If fewer than K pass the threshold, keep only those above threshold.
    - If none pass the threshold, fall back to the single best (max similarity).
    - If at least 1 passes, sample one category according to probabilities
      converted from similarity scores (softmax over kept scores with `temperature`).

    Returns a dict with keys:
      - selected: str
      - kept_categories: List[str]
      - kept_similarities: List[float]
      - probabilities: List[float]
      - all_similarities: List[float] (aligned with input `categories`)
    """
    model, tokenizer, device_str = load_clip_model(model_name=model_name, device=device)
    # Pass (model, tokenizer) to similarity function so it can tokenize
    sims = compute_text_cosine_similarities(word, categories, (model, tokenizer), device_str, normalize=True)

    # Identify indices meeting threshold
    passing = [(i, float(sims[i])) for i in range(len(categories)) if float(sims[i]) >= threshold]

    if len(passing) == 0:
        # Fallback: pick argmax deterministically
        best_idx = int(torch.argmax(sims).item())
        return {
            "selected": categories[best_idx],
            "kept_categories": [categories[best_idx]],
            "kept_similarities": [float(sims[best_idx])],
            "probabilities": [1.0],
            "all_similarities": [float(x) for x in sims.tolist()],
        }

    # Sort passing by similarity descending and take top_k
    passing.sort(key=lambda x: x[1], reverse=True)
    passing = passing[: min(top_k, len(passing))]

    kept_indices = [i for i, _ in passing]
    kept_scores = torch.tensor([s for _, s in passing], dtype=torch.float32)

    # Convert similarities to probabilities using softmax with temperature
    # Numerical stability: subtract max before softmax
    temp = max(1e-6, float(temperature))
    logits = kept_scores / temp
    logits = logits - torch.max(logits)
    probs = torch.softmax(logits, dim=0)

    # Sample one index according to probabilities
    sampled_rel_idx = torch.multinomial(probs, num_samples=1).item()
    sampled_abs_idx = kept_indices[sampled_rel_idx]

    return {
        "selected": categories[sampled_abs_idx],
        "kept_categories": [categories[i] for i in kept_indices],
        "kept_similarities": [float(x) for x in kept_scores.tolist()],
        "probabilities": [float(x) for x in probs.tolist()],
        "all_similarities": [float(x) for x in sims.tolist()],
    }

class ModifyRoomJson:
    def change(self, data, command_args):
        # delete the space in the command_args
        obj_name = command_args[0].replace(' ', '')
        pos = command_args[1].replace(' ', '').split(',')
        rot = command_args[2].replace(' ', '').split(',')
        for obj in data['scene']['room'][0]['children']:
            if obj['ref'] == obj_name:
                obj['pos'] = [float(p) for p in pos]
                obj['rot'] = [float(r) for r in rot]
        return data

    def add(self, data, command_args):
        obj_name = command_args[0].replace(' ', '')
        pos = command_args[1].replace(' ', '').split(',')
        rot = command_args[2].replace(' ', '').split(',')
        with open(os.path.join(MODEL_INFO_PATH, 'model_info_revised.json'), 'r') as f:
            model_info = json.load(f)
        label_to_model = collections.defaultdict(list)
        for m in model_info:
            label_to_model[m["category"].lower().replace(" / ", "/") if m["category"] else 'others'].append(m["model_id"])

        selected_category = select_category_with_clip(obj_name, list(label_to_model.keys()), 0.5)
        selected_instance = random.choice(label_to_model[selected_category['selected']])

        # Check if the object is already in the json file
        has_new_object = True
        for obj in data['furniture']:
            if obj['jid'] == selected_instance:
                has_new_object = False
                new_uid = obj['uid']
                break  
        
        if has_new_object:
            # gather all the uids in the json file
            all_uids_furniture = [obj['uid'] for obj in data['furniture'] if "uid" in obj]
            all_uids_furniture_num_value = sorted([int(obj['uid'].split('/')[0]) for obj in data['furniture'] if "uid" in obj])
            offset = 1
            new_uid = str(all_uids_furniture_num_value[-1] + offset) + '/model'
            while new_uid in all_uids_furniture:
                offset += 1
                new_uid = str(all_uids_furniture_num_value[-1] + offset) + '/model'
            
            # load the obj file and get the length, width, height of the object
            obj_file = os.path.join(MODEL_INFO_PATH, selected_instance, 'raw_model.obj')
            vertices = parse_obj_file(obj_file)
            # Calculate bounding box
            min_coords, max_coords, dimensions = calculate_bounding_box(vertices)
            # get the max value of the x, y, z of the object
            length = max_coords[0] - min_coords[0]
            width = max_coords[1] - min_coords[1]
            height = max_coords[2] - min_coords[2]

            new_furniture = {
                "jid": selected_instance,
                "uid": new_uid,
                "aid": [],
                "category": selected_category['selected'],
                "bbox": [length, width, height],
                "valid": True
            }
            data['furniture'].append(new_furniture)

        mesh_id = 0
        for obj in data['scene']['room'][0]['children']:
            if "furniture" in obj['instanceid']:
                cur_mesh_id = int(obj['instanceid'].split('/')[-1])
                if cur_mesh_id > mesh_id:
                    mesh_id = cur_mesh_id
        new_obj = {
            "ref": new_uid,
            "pos": [float(p) for p in pos],
            "rot": [float(r) for r in rot],
            "scale": [1, 1, 1],
            "instanceid": "furniture/" + str(mesh_id + 1),
            "category_id": selected_category['selected']
        }
        data['scene']['room'][0]['children'].append(new_obj)

        return data

    