import torch
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
import pandas as pd
import os
import signal
from torch.amp import autocast, GradScaler
import time
import torch.autograd
import gc
import sys
from tqdm import tqdm

from module import PCamDataset, QuantumNeuralNetwork, preprocess_data, calculate_accuracy, set_all_seeds, EarlyStopping, MetricsTracker, get_quantum_circuit_metrics, analyze_data_distribution, test_model_output, prepare_balanced_model, save_checkpoint

# Define global variables for tracking training state
global current_epoch, model, optimizer, results, interrupted, new_version

def signal_handler(sig, frame):
    """Handle interruption signal (Ctrl+C)"""
    global interrupted, current_epoch, model, optimizer, results, new_version
    print('\nTraining interrupted! Saving checkpoint before exiting...')
    interrupted = True
    
    if hasattr(sys, 'ps1'):  # If in interactive mode, just set the flag
        return
    
    # Save the checkpoint if we have a model and optimizer
    if model is not None and optimizer is not None and current_epoch > 0:
        checkpoint_path = save_checkpoint(current_epoch, model, optimizer, results, new_version)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    print("Exiting...")
    sys.exit(0)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set all seeds for reproducibility
seed = 42
set_all_seeds(seed)

# Load datasets with limited samples
max_train_samples = 15000
max_valid_samples = 2000
max_test_samples = 3000

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print(f"Loading datasets with {max_train_samples} training samples...")
trainset = PCamDataset('./data/pcam/train/camelyonpatch_level_2_split_train_x.h5', 
                      './data/pcam/train/camelyonpatch_level_2_split_train_y.h5', 
                      transform=transform, 
                      max_samples=max_train_samples)
validset = PCamDataset('./data/pcam/valid/camelyonpatch_level_2_split_valid_x.h5', 
                      './data/pcam/valid/camelyonpatch_level_2_split_valid_y.h5', 
                      transform=transform, 
                      max_samples=max_valid_samples)
testset = PCamDataset('./data/pcam/test/camelyonpatch_level_2_split_test_x.h5', 
                     './data/pcam/test/camelyonpatch_level_2_split_test_y.h5', 
                     transform=transform, 
                     max_samples=max_test_samples)

trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
validloader = DataLoader(validset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)

num_features = 3
num_classes = 1
num_qubits = 3

# Continue from the previous training run
previous_version = 'optimized_fast_conv_regularized_v2'
new_version = 'extended_15k_samples_v1'

# Find the latest checkpoint
def find_latest_checkpoint(checkpoint_dir='./results/checkpoints', version=previous_version):
    # First, check if the final model exists in the results directory
    final_model_path = f'./results/trained_qnn_model_{version}.pth'
    if os.path.exists(final_model_path):
        print(f"Found final model: {final_model_path}")
        
        # Try to determine epoch from the saved model with weights_only=True for better security
        try:
            checkpoint = torch.load(final_model_path, map_location=device, weights_only=True)
            start_epoch = checkpoint.get('epoch', 0)
            print(f"Resuming from final model (Epoch {start_epoch})")
            return final_model_path, start_epoch
        except Exception as e:
            print(f"Error loading with weights_only=True: {e}")
            print("Falling back to weights_only=False for compatibility")
            try:
                checkpoint = torch.load(final_model_path, map_location=device, weights_only=False)
                start_epoch = checkpoint.get('epoch', 0)
                print(f"Resuming from final model (Epoch {start_epoch})")
                return final_model_path, start_epoch
            except Exception as e:
                print(f"Error loading epoch from model: {e}")
                print("Will use epoch 0 as starting point")
                return final_model_path, 0
    
    # If final model doesn't exist, look for checkpoints
    checkpoints = [f for f in os.listdir(checkpoint_dir) 
                  if f.startswith('qnn_checkpoint_epoch_') and version in f]
    
    if not checkpoints:
        print("No checkpoints found. Will start from scratch.")
        return None, 0
    
    # Extract epoch numbers and find the latest
    epochs = [int(cp.split('_')[3]) for cp in checkpoints]
    latest_idx = epochs.index(max(epochs))
    latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[latest_idx])
    start_epoch = epochs[latest_idx]
    
    print(f"Found latest checkpoint: {latest_checkpoint} (Epoch {start_epoch})")
    return latest_checkpoint, start_epoch

def load_model_from_checkpoint(checkpoint_path):
    """Load model and optimizer state from checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint {checkpoint_path} not found!")
        return None, None, None, 0
    
    print(f"Loading checkpoint from {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Create a new model
        model = QuantumNeuralNetwork(num_features, num_classes, 
                                    init_method='structured', seed=seed).to(device)
        
        # Load the state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        optimizer = optim.AdamW(model.parameters(), lr=0.0009, weight_decay=1e-4)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2, eta_min=1e-5
        )
        
        # Load scheduler state if available
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("Loaded scheduler state successfully")
            except Exception as e:
                print(f"Warning: Could not load scheduler state: {e}")
        
        # Get the epoch
        start_epoch = checkpoint['epoch']
        
        print(f"Loaded model state from epoch {start_epoch}")
        return model, optimizer, scheduler, start_epoch
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None, None, None, 0

def load_results(version=previous_version):
    """Load previous training results if available"""
    results_path = f'./results/training_results_{version}.csv'
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        df = df.drop(df.columns[0], axis=1)
        results = df.values.tolist()
        print(f"Loaded {len(results)} previous training results")
        return results
    else:
        print("No previous results found. Starting with empty results.")
        return []

# Main function to continue training
def continue_training():
    global current_epoch, model, optimizer, results, interrupted, version
    
    # Create directories if they don't exist
    os.makedirs('./results/latest', exist_ok=True)
    os.makedirs('./results/checkpoints', exist_ok=True)
    os.makedirs(f'./results/metrics/qnn_{new_version}_{time.strftime("%Y%m%d_%H%M%S")}', exist_ok=True)
    print("checka")
    
    # Set the version for the current run
    version = new_version
    
    # Find latest checkpoint and load model
    latest_checkpoint, checkpoint_epoch = find_latest_checkpoint()
    
    if latest_checkpoint:
        model, optimizer, scheduler, start_epoch = load_model_from_checkpoint(latest_checkpoint)
        if model is None:
            print("Failed to load checkpoint properly. Starting with a new model.")
            model = QuantumNeuralNetwork(num_features, num_classes, 
                                        init_method='structured', seed=seed).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=0.0009, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=5, T_mult=2, eta_min=1e-5
            )
            start_epoch = 0
            previous_results = []
        else:
            # Load previous results
            previous_results = load_results()
    else:
        # Create new model if no checkpoint found
        model = QuantumNeuralNetwork(num_features, num_classes, 
                                    init_method='structured', seed=seed).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.0009, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2, eta_min=1e-5
        )
        start_epoch = 0
        previous_results = []
    
    # Set the initial current epoch
    current_epoch = start_epoch
    
    # Initialize results with previous results
    results = previous_results.copy()
    print(results)
    
    # Verify model outputs for numerical stability
    model = prepare_balanced_model(trainloader, model, device=device)
    
    # Define criterion (loss function)
    criterion = nn.BCEWithLogitsLoss()
    
    # Data analysis
    print("\n=== DATA ANALYSIS ===")
    label_dist, _ = analyze_data_distribution(trainloader, max_samples=10)
    print("\n=== INITIAL MODEL OUTPUT ===")
    initial_outputs = test_model_output(model, trainloader, device=device)
    
    # Adjust loss function based on class imbalance
    if abs(label_dist[0] - label_dist[1]) > 0.1 * sum(label_dist.values()):
        print("Class imbalance detected! Using weighted loss...")
        pos_weight = torch.tensor([label_dist[0]/max(1, label_dist[1])]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Using positive class weight: {pos_weight.item():.4f}")
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Define training parameters
    epochs = 50  # Additional epochs to train
    validation_frequency = 2
    max_eval_batches = 10
    test_frequency = 10
    
    # Initialize gradient scaler for mixed precision
    scaler = GradScaler()
    
    # Initialize metrics tracker with new version
    metrics_tracker = MetricsTracker(f"qnn_{new_version}_{time.strftime('%Y%m%d_%H%M%S')}")
    
    early_stopping = EarlyStopping(patience=5, monitor_test=True)
    
    # Global variables for training state
    interrupted = False
    
    # Main training loop
    print(f"\n=== CONTINUING TRAINING FROM EPOCH {start_epoch+1} ===")
    print(f"Training for {epochs} additional epochs with {max_train_samples} samples")
    
    for epoch in range(epochs):
        current_epoch = start_epoch + epoch + 1
        model.train()
        running_loss = 0.0
        batch_start_time = time.time()
        
        # Start timing the epoch
        metrics_tracker.start_epoch(current_epoch)
        
        if interrupted:
            break
            
        # Progress bar for current epoch
        with tqdm(total=len(trainloader), desc=f"Epoch [{current_epoch}/{start_epoch+epochs}]", ncols=100) as progress_bar:
            for i, data in enumerate(trainloader, 0):
                if interrupted:
                    break
                    
                inputs, labels = data
                inputs, labels = preprocess_data(inputs, labels)
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                
                # Mixed precision training
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = model(inputs)
                    loss = criterion(outputs.squeeze(), labels)
                    
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Debug output
                if i % 10 == 0:
                    print(f"Batch {i}: Loss={loss.item():.4f}")
                
                # Monitor gradients
                grad_norm = None
                if i % 50 == 0:
                    scaler.unscale_(optimizer)
                    grad_norm = log_gradients(model, epoch * len(trainloader) + i)
                    
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()

                # Track batch metrics
                batch_time = time.time() - batch_start_time
                current_lr = optimizer.param_groups[0]['lr']
                metrics_tracker.log_batch(current_epoch, i, loss.item(), current_lr, batch_time, grad_norm)
                batch_start_time = time.time()

                # Clear memory
                if hasattr(model, 'quantum_layer'):
                    model.quantum_layer.zero_grad(set_to_none=True)
                    
                running_loss += loss.item()

                # Update progress bar
                progress_bar.set_postfix({"Loss": running_loss / (i + 1)})
                progress_bar.update(1)
                
                # Memory management
                del inputs, labels, outputs, loss
                if i % 10 == 0:
                    gc.collect()
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None

        if interrupted:
            break

        # Calculate training metrics
        train_accuracy, train_loss, train_metrics = calculate_accuracy(
            trainloader, model, device=device, criterion=criterion, max_batches=max_eval_batches, phase='train')
        
        # Log detailed training metrics
        metrics_tracker.log_detailed_metrics(current_epoch, 'train', train_metrics)
        
        # Validation
        if current_epoch % validation_frequency == 0:
            val_accuracy, val_loss, val_metrics = calculate_accuracy(
                validloader, model, device=device, criterion=criterion, max_batches=max_eval_batches, phase='val')
            
            print(f"Epoch [{current_epoch}/{start_epoch+epochs}] - "
                  f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
                  f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
            
            # Log validation metrics
            metrics_tracker.log_detailed_metrics(current_epoch, 'val', val_metrics)
            
            # Test evaluation
            test_metrics = None
            if current_epoch % test_frequency == 0:
                test_accuracy, test_loss, test_metrics = calculate_accuracy(
                    testloader, model, device=device, criterion=criterion, max_batches=max_eval_batches, phase='test')
                print(f"Epoch {current_epoch} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
                metrics_tracker.log_detailed_metrics(current_epoch, 'test', test_metrics)
            
            # Save epoch metrics
            metrics_tracker.log_epoch(current_epoch, train_loss, train_accuracy, val_loss, val_accuracy)
            
            # Update results list
            results.append([current_epoch, running_loss / len(trainloader), train_loss, train_accuracy, val_loss, val_accuracy])
            
            # Quantum metrics
            quantum_metrics = get_quantum_circuit_metrics(model)
            metrics_tracker.log_quantum_metrics(current_epoch, quantum_metrics)
            
            # Early stopping check
            early_stopping(val_loss, test_metrics)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
                
            # Update learning rate scheduler
            scheduler.step(val_loss)
        else:
            print(f"Epoch [{current_epoch}/{start_epoch+epochs}] - "
                  f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
            
            # For epochs without validation
            metrics_tracker.log_epoch(current_epoch, train_loss, train_accuracy)
            results.append([current_epoch, running_loss / len(trainloader), train_loss, train_accuracy, None, None])
            
            # Update scheduler with training loss
            scheduler.step(train_loss)

        # Save checkpoint periodically
        if current_epoch % 10 == 0:  # Save every 10 epochs
            save_checkpoint(current_epoch, model, optimizer, results, new_version)
            print(f"Checkpoint saved at epoch {current_epoch}")

    # Save final model and results
    if not interrupted:
        print("Extended training completed successfully. Saving final model and results...")
    else:
        print("Training interrupted. Saving current state...")

    # Save final model
    save_checkpoint(current_epoch, model, optimizer, results, new_version, final=True)

    # Final evaluation
    if not interrupted:
        print("\n=== FINAL MODEL EVALUATION ===")
        test_accuracy, test_loss, test_metrics = calculate_accuracy(testloader, model, device=device, criterion=criterion, detailed=True, phase='test')
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
        
        # Save test metrics
        metrics_tracker.save_test_metrics(test_metrics)
        
        # Generate summary
        summary = metrics_tracker.save_all()
        print("\n=== EXPERIMENT SUMMARY ===")
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        # Final output distribution
        print("\n=== FINAL MODEL OUTPUT DISTRIBUTION ===")
        final_outputs = test_model_output(model, testloader, device=device)

    print("Extended training completed!")
    
    # Display model configuration
    print("\n=== QNN CONFIGURATION ===")
    print(f"Using pure QNN with {num_qubits} qubits")
    print("Quantum circuit structure:")
    print(model.qnn.circuit.draw())

    # Circuit parameters
    print("\n=== QNN CIRCUIT PARAMETERS ===")
    total_qubits = num_qubits
    total_rotations = 0
    total_cnots = 0

    for gate in model.qnn.circuit.data:
        if gate.operation.name in ['rx', 'ry', 'rz', 'u3']:
            total_rotations += 1
        elif gate.operation.name in ['cx']:
            total_cnots += 1

    print(f"Circuit complexity: {total_qubits} qubits, {total_rotations} rotations, {total_cnots} CNOT gates")
    print(f"Total trainable parameters in quantum circuit: {len(list(model.ansatz.parameters))}")
    print(f"Total trainable parameters in full model: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

def log_gradients(model, step):
    """Log gradient statistics to help debug training issues"""
    total_norm = 0
    param_norms = []
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            param_norms.append(param_norm.item())
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    print(f"Step {step} - Gradient norm: {total_norm:.5f}")
    if total_norm < 1e-4:
        print("WARNING: Gradients are very small, model might not be learning!")
    elif total_norm > 0.5:
        print("INFO: Gradient norm is relatively high")
    
    if param_norms:
        print(f"  Min/Max grad: {min(param_norms):.5f}/{max(param_norms):.5f}")
        
    return total_norm

def save_checkpoint(epoch, model, optimizer, results_so_far, version, final=False):
    """Save model checkpoint and training results"""
    checkpoint_dir = './results/checkpoints'
    if final:
        checkpoint_path = f'./results/trained_qnn_model_{version}.pth'
    else:
        checkpoint_path = f'{checkpoint_dir}/qnn_checkpoint_epoch_{epoch}_{version}.pth'
    
    # Save model and optimizer state
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'version': version,
        'timestamp': time.time(),  # Add timestamp for better tracking
    }, checkpoint_path)
    
    # Save training results so far
    if results_so_far:
        df = pd.DataFrame(results_so_far, columns=['Epoch', 'Running Loss', 'Train Loss', 'Train Accuracy', 'Val Loss', 'Val Accuracy'])
        if final:
            df.to_csv(f'./results/training_results_{version}.csv', index=True)
        else:
            df.to_csv(f'{checkpoint_dir}/training_results_checkpoint_{version}.csv', index=True)
    
    return checkpoint_path

if __name__ == "__main__":
    print("Starting extended training...")
    continue_training()
