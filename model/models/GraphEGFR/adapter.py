from collections.abc import Mapping


class GraphEGFRAdapter:
    """Load GraphEGFR and run inference on a prepared batch."""

    def __init__(
        self, path_weights=None, device="cpu", *, trusted_checkpoint=False,
        **model_kwargs,
    ):
        from .graphegfr.models import GraphEGFR

        self.device = device
        if path_weights is None:
            self.model = GraphEGFR(**model_kwargs).to(device).eval()
        else:
            self.load_weights(path_weights, trusted_checkpoint=trusted_checkpoint, **model_kwargs)

    def load_weights(self, path, *, trusted_checkpoint=False, **model_kwargs):
        """Load a state dict or an explicitly trusted model checkpoint."""
        import torch

        from .graphegfr.models import GraphEGFR

        checkpoint = torch.load(path, map_location="cpu", weights_only=not trusted_checkpoint)
        if isinstance(checkpoint, (list, tuple)):
            checkpoint = checkpoint[0]  # Saver in graphegfr stores [model, optimizer].

        if isinstance(checkpoint, GraphEGFR) and trusted_checkpoint:
            model = checkpoint
        elif isinstance(checkpoint, Mapping):
            state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
            state = {key.removeprefix("module."): value for key, value in state.items()}
            if not model_kwargs and hasattr(self, "model"):
                model = self.model
            else:
                model_kwargs.setdefault("fingfeaf", state["dnn1.weight"].shape[1])
                model_kwargs.setdefault("fingfeac", state["cnn1.weight"].shape[1])
                model = GraphEGFR(**model_kwargs)
            model.load_state_dict(state)
        else:
            raise ValueError("Expected GraphEGFR weights or a trusted GraphEGFR checkpoint")

        self.model = model.to(self.device).eval()

    def predict(self, batch):
        """Return predictions for a batch already prepared on the model device."""
        import torch

        self.model.eval()
        with torch.inference_mode():
            return self.model(batch)
