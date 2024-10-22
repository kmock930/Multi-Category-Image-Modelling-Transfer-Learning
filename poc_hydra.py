import hydra
from omegaconf import DictConfig, OmegaConf

# Proof of concept: Using Hydra to access configurations
@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    # Print original configuration
    print("Model Configuration:")
    print(f"Type: {cfg.model.type}")
    print(f"Hidden Units: {cfg.model.hidden_units}")
    print(f"Dropout Rate: {cfg.model.dropout_rate}")
    print(f"Activation Function: {cfg.model.activation}")
    
    print("\nTraining Configuration:")
    print(f"Learning Rate: {cfg.training.learning_rate}")
    print(f"Batch Size: {cfg.training.batch_size}")
    print(f"Epochs: {cfg.training.epochs}")
    print(f"Optimizer: {cfg.training.optimizer}")
    
    print("\nData Augmentation:")
    print(f"Flip: {cfg.data.augmentation.flip}")
    print(f"Rotation Range: {cfg.data.augmentation.rotation_range}")
    print(f"Zoom Range: {cfg.data.augmentation.zoom_range}")

    # Modify configuration values during runtime
    cfg.training.learning_rate = 0.0005
    cfg.training.batch_size = 64

    # Print modified values
    print("Modified Learning Rate:", cfg.training.learning_rate)
    print("Modified Batch Size:", cfg.training.batch_size)

if __name__ == "__main__":
    main()