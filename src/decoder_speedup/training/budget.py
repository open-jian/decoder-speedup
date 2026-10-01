"""Resolve a fixed dataset exposure budget into complete optimizer updates."""
from ..config import positive_int


def resolve_budget(config, rows):
    config.validate()
    positive_int(rows, "training manifest rows")
    t = config.training
    effective_batch = config.data.batch_size * t.accumulation
    plan = {"effective_batch": effective_batch, "training_rows": rows,
            "world_size": 1, "lr_schedule": t.lr_schedule}
    if t.reconstruction_epochs is not None:
        samples = rows * t.reconstruction_epochs
        updates = (samples + effective_batch - 1) // effective_batch
        plan.update(basis="epochs", reconstruction_epochs=t.reconstruction_epochs,
                    requested_reconstruction_samples=samples,
                    rounded_extra_samples=updates * effective_batch - samples)
        t.reconstruction_epochs = None
        t.reconstruction_updates = updates
    else:
        plan["basis"] = "updates"
    plan.update(reconstruction_updates=t.reconstruction_updates, adversarial_updates=t.adversarial_updates)
    config.validate()
    return plan
