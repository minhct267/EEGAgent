import torch
def multi_model_predict(models, data_tensor, mask=None):
    """Average softmax scores across models (soft voting)."""
    device = next(models[0].parameters()).device
    data_tensor = data_tensor.to(device)

    all_probs = []
    with torch.no_grad():
        for model in models:
            if mask is not None:
                logits = model(data_tensor, mask.to(device))
            else:
                logits = model(data_tensor)

            probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs.cpu())

    all_probs = torch.stack(all_probs)  # shape: (num_models, 1, num_classes)
    avg_probs = all_probs.mean(dim=0)  # shape: (1, num_classes)
    return avg_probs.numpy().flatten()