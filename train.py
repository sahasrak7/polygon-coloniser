import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

# Import based on which data_utils version you're using
# Option 1: If using the PyTorch transforms version (recommended)
try:
    from scripts.data_utils import create_datasets
    USE_PYTORCH_TRANSFORMS = True
    print("Using PyTorch transforms version")
except ImportError:
    # Option 2: If using the Albumentations version
    from scripts.data_utils import PolygonDataset, train_transform, val_transform
    USE_PYTORCH_TRANSFORMS = False
    print("Using Albumentations transforms version")

from models.unet import UNet

# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_MODEL_PATH = 'checkpoints/unet_model.pth'

# Create checkpoints directory if it doesn't exist
os.makedirs('checkpoints', exist_ok=True)

print(f"Using device: {DEVICE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Epochs: {EPOCHS}")

# Initialize wandb
wandb.init(project="ayna_ml_assignment", config={
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "epochs": EPOCHS,
    "device": str(DEVICE),
    "model": "UNet",
    "transforms": "PyTorch" if USE_PYTORCH_TRANSFORMS else "Albumentations"
})

# Create datasets and dataloaders
try:
    if USE_PYTORCH_TRANSFORMS:
        # Using PyTorch transforms version
        train_dataset, val_dataset = create_datasets(
            'data/training', 'data/training/data.json',
            'data/validation', 'data/validation/data.json'
        )
    else:
        # Using Albumentations version
        train_dataset = PolygonDataset('data/training', 'data/training/data.json', transform=train_transform)
        val_dataset = PolygonDataset('data/validation', 'data/validation/data.json', transform=val_transform)
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
except Exception as e:
    print(f"Error creating datasets: {e}")
    print("Make sure your data directories and JSON files exist:")
    print("- data/training/data.json")
    print("- data/validation/data.json")
    raise

# Create data loaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=2 if DEVICE.type == 'cuda' else 0,
    pin_memory=True if DEVICE.type == 'cuda' else False
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    num_workers=2 if DEVICE.type == 'cuda' else 0,
    pin_memory=True if DEVICE.type == 'cuda' else False
)

# Test data loading
print("Testing data loading...")
try:
    sample_batch = next(iter(train_loader))
    input_img, color_idx, target_img = sample_batch
    print(f"Input image shape: {input_img.shape}")
    print(f"Color index shape: {color_idx.shape}")
    print(f"Target image shape: {target_img.shape}")
    print("Data loading test successful!")
except Exception as e:
    print(f"Data loading test failed: {e}")
    raise

# Model, Loss, Optimizer
model = UNet(in_channels=65, out_channels=3).to(DEVICE)  # 1 (grayscale) + 64 (color embedding)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Model summary
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Training Loop
best_val_loss = float('inf')
train_losses = []
val_losses = []

print("\nStarting training...")
for epoch in range(EPOCHS):
    # Training phase
    model.train()
    train_loss = 0
    train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Train]')
    
    for batch_idx, (input_img, color_idx, target_img) in enumerate(train_pbar):
        try:
            input_img, color_idx, target_img = input_img.to(DEVICE), color_idx.to(DEVICE), target_img.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(input_img, color_idx)
            loss = criterion(output, target_img)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        except Exception as e:
            print(f"Error in training batch {batch_idx}: {e}")
            print(f"Input shape: {input_img.shape if 'input_img' in locals() else 'Unknown'}")
            print(f"Color shape: {color_idx.shape if 'color_idx' in locals() else 'Unknown'}")
            print(f"Target shape: {target_img.shape if 'target_img' in locals() else 'Unknown'}")
            raise
    
    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # Validation phase
    model.eval()
    val_loss = 0
    val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Val]')
    
    with torch.no_grad():
        for batch_idx, (input_img, color_idx, target_img) in enumerate(val_pbar):
            try:
                input_img, color_idx, target_img = input_img.to(DEVICE), color_idx.to(DEVICE), target_img.to(DEVICE)
                output = model(input_img, color_idx)
                loss = criterion(output, target_img)
                val_loss += loss.item()
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            except Exception as e:
                print(f"Error in validation batch {batch_idx}: {e}")
                raise
    
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    # Update learning rate scheduler
    scheduler.step(avg_val_loss)
    
    # Log to wandb
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "learning_rate": optimizer.param_groups[0]['lr']
    })
    
    # Print epoch results
    print(f'Epoch {epoch+1}/{EPOCHS}:')
    print(f'  Train Loss: {avg_train_loss:.6f}')
    print(f'  Val Loss: {avg_val_loss:.6f}')
    print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.8f}')
    
    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'best_val_loss': best_val_loss
        }, SAVE_MODEL_PATH)
        print(f'  New best model saved! Val Loss: {best_val_loss:.6f}')
    
    print('-' * 50)

# Final model save
final_model_path = 'checkpoints/unet_model_final.pth'
torch.save({
    'epoch': EPOCHS,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_losses': train_losses,
    'val_losses': val_losses,
    'best_val_loss': best_val_loss
}, final_model_path)

print(f"\nTraining completed!")
print(f"Best validation loss: {best_val_loss:.6f}")
print(f"Best model saved to: {SAVE_MODEL_PATH}")
print(f"Final model saved to: {final_model_path}")

# Upload models to wandb
wandb.save(SAVE_MODEL_PATH)
wandb.save(final_model_path)

# Create a simple training curve plot
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='blue')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_curve.png', dpi=300, bbox_inches='tight')
wandb.log({"training_curve": wandb.Image('training_curve.png')})

wandb.finish()
print("Training script completed successfully!")