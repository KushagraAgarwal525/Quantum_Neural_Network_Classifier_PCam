import h5py
from torch.utils.data import Dataset
from torch import nn, optim
import torch
import numpy as np
import random
from qiskit import transpile
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit import QuantumCircuit
import sys
import pandas as pd
import os
import time


class QuantumParameterInitializer:
    """Handles initialization of quantum circuit parameters."""
    
    @staticmethod
    def zero_init(num_params):
        """Initialize all parameters to zero."""
        return np.zeros(num_params)
    
    @staticmethod
    def uniform_init(num_params, low=0.0, high=0.1):
        """Initialize with uniform small values to avoid extreme rotations."""
        return np.random.uniform(low=low, high=high, size=num_params) * np.pi
    
    @staticmethod
    def balanced_init(num_params):
        """Initialize with balanced values distributed around key rotation points."""
        # Use multiples of π/4 for better quantum state coverage
        base_points = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4])
        # Choose random base points and add small noise
        indices = np.random.choice(len(base_points), num_params)
        points = base_points[indices]
        noise = np.random.uniform(-0.1, 0.1, num_params) * np.pi
        return points + noise
    
    @staticmethod
    def hardware_efficient_init(num_params):
        """Initialize with small values suitable for hardware efficient ansatze."""
        return (np.random.random(num_params) - 0.5) * 0.1 * np.pi
    
    @staticmethod
    def structured_init(num_params):
        """Layered initialization to address barren plateau issues.
        
        Initializes parameters with progressively larger ranges for deeper layers,
        which helps prevent vanishing gradients in quantum circuits.
        """
        init_params = []
        # Estimate number of layers based on parameter count
        num_layers = max(3, num_params // 3)  # At least 3 layers
        params_per_layer = max(1, num_params // num_layers)
        
        remaining_params = num_params
        
        # First layer: very small values [-0.05π, 0.05π] for better convergence
        layer_size = min(params_per_layer, remaining_params)
        init_params.extend(np.random.uniform(-0.05, 0.05, layer_size) * np.pi)
        remaining_params -= layer_size
        
        # Middle layers: gradually increasing ranges
        for i in range(1, num_layers-1):
            if remaining_params <= 0:
                break
                
            # Scale increases with layer depth
            scale = 0.1 + 0.4 * (i / (num_layers-1))
            layer_size = min(params_per_layer, remaining_params)
            init_params.extend(np.random.uniform(-scale, scale, layer_size) * np.pi)
            remaining_params -= layer_size
        
        # Final layer: wider values for better expressivity [-0.7π, 0.7π]
        if remaining_params > 0:
            init_params.extend(np.random.uniform(-0.7, 0.7, remaining_params) * np.pi)
        
        return np.array(init_params)

class EarlyStopping:
    def __init__(self, patience=5, delta=0, monitor_test=False):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.best_test_score = None
        self.early_stop = False
        self.counter = 0
        self.monitor_test = monitor_test
        
    def __call__(self, val_loss, test_metrics=None):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if test_metrics:
                self.best_test_score = test_metrics['accuracy']
        elif score < self.best_score + self.delta:
            self.counter += 1
            # If test accuracy drops, stop earlier
            if self.monitor_test and test_metrics and hasattr(self, 'best_test_score'):
                if test_metrics['accuracy'] < self.best_test_score - 1.0:  # Stop if test accuracy drops by more than 1%
                    self.counter += 1  # Accelerate early stopping
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            if test_metrics:
                self.best_test_score = test_metrics['accuracy']


class QuantumCircuitCache:
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.cache = {}
        self.keys = []

    def get(self, inputs_tensor):
        # Convert inputs to hashable format and round to reduce variations
        key = self._tensor_to_hashable(inputs_tensor)
        return self.cache.get(key, None)

    def put(self, inputs_tensor, outputs):
        key = self._tensor_to_hashable(inputs_tensor)
        if key not in self.cache:
            if len(self.keys) >= self.capacity:
                # Remove oldest entry
                old_key = self.keys.pop(0)
                del self.cache[old_key]
            self.keys.append(key)
        self.cache[key] = outputs

    def _tensor_to_hashable(self, tensor):
        # Create a hashable representation of the tensor without modifying the original
        # Round to reduce minor variations and convert to tuple for hashing
        return tuple(np.round(tensor.detach().cpu().numpy().flatten(), decimals=2))

class MetricsTracker:
    """Track and store detailed metrics for ML research paper analysis"""
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.output_dir = f'./results/metrics/{experiment_name}'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize dictionaries to store all metrics
        self.training_metrics = {
            'epoch': [], 'batch': [], 'loss': [], 'lr': [], 
            'batch_time': [], 'gradient_norm': []
        }
        
        self.epoch_metrics = {
            'epoch': [], 'train_loss': [], 'train_accuracy': [], 
            'val_loss': [], 'val_accuracy': [], 'epoch_time': []
        }
        
        self.detailed_metrics = {
            'epoch': [], 'phase': [], 'accuracy': [], 'loss': [],
            'precision': [], 'recall': [], 'f1': [], 
            'tn': [], 'fp': [], 'fn': [], 'tp': []
        }
        
        # Track quantum-specific metrics
        self.quantum_metrics = {
            'epoch': [], 'circuit_depth': [], 'num_gates': [],
            'num_parameters': [], 'execution_time': []
        }
        
        # To store final test results
        self.test_metrics = {}
        
        # Start time tracking
        self.start_time = time.time()
        self.epoch_start_time = None
    
    def start_epoch(self, epoch):
        """Mark the start of an epoch for timing purposes"""
        self.epoch_start_time = time.time()
        
    def log_batch(self, epoch, batch, loss, lr, batch_time, gradient_norm=None):
        """Log metrics for a single batch"""
        self.training_metrics['epoch'].append(epoch)
        self.training_metrics['batch'].append(batch)
        self.training_metrics['loss'].append(loss)
        self.training_metrics['lr'].append(lr)
        self.training_metrics['batch_time'].append(batch_time)
        self.training_metrics['gradient_norm'].append(gradient_norm if gradient_norm else float('nan'))
        
        # Save periodically to avoid data loss on crash
        if len(self.training_metrics['epoch']) % 100 == 0:
            self.save_batch_metrics()
    
    def log_epoch(self, epoch, train_loss, train_accuracy, val_loss=None, val_accuracy=None):
        """Log summary metrics for an epoch"""
        epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0
        
        self.epoch_metrics['epoch'].append(epoch)
        self.epoch_metrics['train_loss'].append(train_loss)
        self.epoch_metrics['train_accuracy'].append(train_accuracy)
        self.epoch_metrics['val_loss'].append(val_loss if val_loss is not None else float('nan'))
        self.epoch_metrics['val_accuracy'].append(val_accuracy if val_accuracy is not None else float('nan'))
        self.epoch_metrics['epoch_time'].append(epoch_time)
        
        # Save after each epoch
        self.save_epoch_metrics()
    
    def log_detailed_metrics(self, epoch, phase, metrics_dict):
        """Log detailed metrics including confusion matrix data"""
        self.detailed_metrics['epoch'].append(epoch)
        self.detailed_metrics['phase'].append(phase)
        self.detailed_metrics['accuracy'].append(metrics_dict.get('accuracy', float('nan')))
        self.detailed_metrics['loss'].append(metrics_dict.get('loss', float('nan')))
        self.detailed_metrics['precision'].append(metrics_dict.get('precision', float('nan')))
        self.detailed_metrics['recall'].append(metrics_dict.get('recall', float('nan')))
        self.detailed_metrics['f1'].append(metrics_dict.get('f1', float('nan')))
        self.detailed_metrics['tn'].append(metrics_dict.get('tn', float('nan')))
        self.detailed_metrics['fp'].append(metrics_dict.get('fp', float('nan')))
        self.detailed_metrics['fn'].append(metrics_dict.get('fn', float('nan')))
        self.detailed_metrics['tp'].append(metrics_dict.get('tp', float('nan')))
        
        # Save detailed metrics
        self.save_detailed_metrics()
    
    def log_quantum_metrics(self, epoch, circuit_metrics):
        """Log quantum circuit specific metrics"""
        self.quantum_metrics['epoch'].append(epoch)
        self.quantum_metrics['circuit_depth'].append(circuit_metrics.get('depth', 0))
        self.quantum_metrics['num_gates'].append(circuit_metrics.get('num_gates', 0))
        self.quantum_metrics['num_parameters'].append(circuit_metrics.get('num_parameters', 0))
        self.quantum_metrics['execution_time'].append(circuit_metrics.get('execution_time', 0))
        
        # Save quantum metrics
        self.save_quantum_metrics()
    
    def save_test_metrics(self, test_metrics):
        """Save final test metrics"""
        self.test_metrics = test_metrics
        
        # Save as JSON
        with open(f'{self.output_dir}/test_metrics.json', 'w') as f:
            import json
            json.dump(test_metrics, f, indent=2)
    
    def save_batch_metrics(self):
        """Save batch-level training metrics to CSV"""
        df = pd.DataFrame(self.training_metrics)
        df.to_csv(f'{self.output_dir}/batch_metrics.csv', index=False)
    
    def save_epoch_metrics(self):
        """Save epoch-level metrics to CSV"""
        df = pd.DataFrame(self.epoch_metrics)
        df.to_csv(f'{self.output_dir}/epoch_metrics.csv', index=False)
    
    def save_detailed_metrics(self):
        """Save detailed metrics to CSV"""
        df = pd.DataFrame(self.detailed_metrics)
        df.to_csv(f'{self.output_dir}/detailed_metrics.csv', index=False)
    
    def save_quantum_metrics(self):
        """Save quantum-specific metrics to CSV"""
        df = pd.DataFrame(self.quantum_metrics)
        df.to_csv(f'{self.output_dir}/quantum_metrics.csv', index=False)
    
    def save_all(self):
        """Save all metrics and generate summary"""
        self.save_batch_metrics()
        self.save_epoch_metrics()
        self.save_detailed_metrics()
        self.save_quantum_metrics()
        
        # Generate summary
        total_time = time.time() - self.start_time
        
        # Extract valid validation accuracies (non-NaN values)
        val_accuracies = np.array(self.epoch_metrics['val_accuracy'])
        valid_indices = ~np.isnan(val_accuracies)
        
        # Only proceed if we have some valid values
        if np.any(valid_indices):
            best_val_idx = np.nanargmax(val_accuracies)
            best_val_accuracy = val_accuracies[best_val_idx]
            best_val_epoch = self.epoch_metrics['epoch'][best_val_idx]
        else:
            best_val_accuracy = None
            best_val_epoch = None
            
        summary = {
            'experiment_name': self.experiment_name,
            'total_time_seconds': total_time,
            'total_time_formatted': f"{total_time//3600:.0f}h {(total_time%3600)//60:.0f}m {total_time%60:.0f}s",
            'num_epochs': max(self.epoch_metrics['epoch']) if self.epoch_metrics['epoch'] else 0,
            'best_val_accuracy': best_val_accuracy,
            'best_val_epoch': best_val_epoch,
            'final_test_accuracy': self.test_metrics.get('accuracy', None),
            'final_test_f1': self.test_metrics.get('f1', None),
        }
        
        # Save summary as JSON
        with open(f'{self.output_dir}/summary.json', 'w') as f:
            import json
            json.dump(summary, f, indent=2)
            
        return summary


class PCamDataset(Dataset):
    def __init__(self, h5_file_x, h5_file_y, transform=None, max_samples=None):
        self.h5_file_x = h5_file_x
        self.h5_file_y = h5_file_y
        self.transform = transform
        # Load data lazily only when needed
        with h5py.File(h5_file_x, 'r') as f:
            self.num_samples = len(f['x'])
            # Limit number of samples if specified
            if max_samples and max_samples < self.num_samples:
                self.num_samples = max_samples
        
        # Store file handles instead of loading all data
        self.file_x = None
        self.file_y = None
        self.labels = None
    
    def _open_files(self):
        if self.file_x is None:
            self.file_x = h5py.File(self.h5_file_x, 'r')
            self.file_y = h5py.File(self.h5_file_y, 'r')
            self.labels = self.file_y['y'][:self.num_samples].astype(int)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        self._open_files()
        img = self.file_x['x'][idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label
    
    def __del__(self):
        try:
            if hasattr(self, 'file_x') and self.file_x is not None:
                self.file_x.close()
            if hasattr(self, 'file_y') and self.file_y is not None:
                self.file_y.close()
        except Exception:
            pass  # Suppress errors during garbage collection

class QuantumNeuralNetwork(nn.Module):
    def __init__(self, num_features, num_classes, init_method='hardware_efficient', seed=42):
        super().__init__()
        # Set seed before creating quantum components
        self.seed = seed
        set_all_seeds(self.seed)
        
        # Store initialization method
        self.init_method = init_method
        
        # Create quantum components
        self.qnn, self.ansatz = create_qnn(num_features)
        self.quantum_layer = TorchConnector(self.qnn)
        self.classifier = nn.Linear(2, num_classes, bias=True)
        self.cache = QuantumCircuitCache(capacity=500)  # Cache for quantum circuit outputs
        
        # Initialize both quantum and classical parameters
        self.initialize_parameters()

    def initialize_parameters(self):
        """Initialize both quantum and classical parameters with controlled randomness."""
        # Get number of quantum parameters
        if hasattr(self.qnn, 'weights'):
            num_params = len(self.qnn.weights)
        elif hasattr(self.ansatz, 'parameters'):
            num_params = len(self.ansatz.parameters)
        else:
            num_params = 0
            print("Warning: Could not determine number of quantum parameters")
        
        # Initialize quantum parameters based on chosen method
        if num_params > 0:
            set_all_seeds(self.seed)  # Ensure consistent initialization
            
            # Use structured initialization by default to address barren plateaus
            if self.init_method == 'structured':
                params = QuantumParameterInitializer.structured_init(num_params)
            elif self.init_method == 'hardware_efficient':
                params = QuantumParameterInitializer.hardware_efficient_init(num_params)
            elif self.init_method == 'balanced':
                params = QuantumParameterInitializer.balanced_init(num_params)
            else:
                params = QuantumParameterInitializer.structured_init(num_params)
            
            # Set quantum parameters if possible
            if hasattr(self.qnn, 'set_weights'):
                self.qnn.set_weights(params)
            
        # Initialize classical parameters with non-zero bias to improve early convergence
        with torch.no_grad():
            torch.manual_seed(self.seed + 1)
            nn.init.xavier_uniform_(self.classifier.weight, gain=1.4)  # Higher gain
            self.classifier.bias.fill_(0.1)  # Start with small positive bias
    
    def verify_output_distribution(self, sample_inputs, target_mean=0.0, max_attempts=5):
        """Verify and correct output distribution if too biased."""
        self.eval()
        with torch.no_grad():
            # Get initial outputs
            outputs = self(sample_inputs)
            current_mean = outputs.mean().item()
            current_std = outputs.std().item()
            
            print(f"Initial output distribution - Mean: {current_mean:.4f}, STD: {current_std:.4f}")
            
            # If mean is far from target, adjust the bias to compensate
            if abs(current_mean - target_mean) > 0.3:
                # Calculate bias correction
                bias_correction = target_mean - current_mean
                print(f"Adjusting bias by {bias_correction:.4f} to center outputs")
                self.classifier.bias.add_(bias_correction)
                
                # Verify the correction worked
                new_outputs = self(sample_inputs)
                new_mean = new_outputs.mean().item()
                print(f"After bias correction - Mean: {new_mean:.4f}")
                
            # If variance is too low, try re-initializing with different seed
            attempt = 0
            while current_std < 0.05 and attempt < max_attempts:
                attempt += 1
                print(f"Output variance too low ({current_std:.4f}). Attempt {attempt}/{max_attempts} to re-initialize.")
                # Try a different seed
                self.seed += 100
                self.initialize_parameters()
                outputs = self(sample_inputs)
                current_std = outputs.std().item()
                print(f"New initialization - STD: {current_std:.4f}")

    def forward(self, x):
        # Process batch - create a fixed output tensor
        batch_size = x.size(0)
        outputs_list = []
        
        # Process each input separately to leverage caching
        for i in range(batch_size):
            # Make a deep copy to avoid in-place modifications of the input
            single_input = x[i:i+1].clone().detach().requires_grad_(x.requires_grad)
            cached_output = self.cache.get(single_input)
            
            if cached_output is not None:
                # Create a fresh tensor from the cached output to avoid gradient issues
                outputs_list.append(cached_output.clone().detach().requires_grad_(x.requires_grad))
            else:
                # Compute quantum output if not in cache
                quantum_output = self.quantum_layer(single_input)
                # Store a detached copy in the cache
                self.cache.put(single_input, quantum_output.clone().detach())
                outputs_list.append(quantum_output)
                
        # Stack outputs along batch dimension without using in-place operations
        x = torch.cat(outputs_list, dim=0)
        # Removed dropout application to maintain pure QNN
        return self.classifier(x).squeeze()

def set_all_seeds(seed=42):
    """Set seeds for all random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Return seed in case it's dynamically generated
    return seed

def create_qnn(num_features):
    feature_map = ZZFeatureMap(num_features, reps=1, entanglement="linear")
    
    ansatz = TwoLocal(num_features, 
                     rotation_blocks=["ry"],  # Keep single rotation type for speed
                     entanglement_blocks="cx", 
                     entanglement="linear",   # Linear entanglement is faster
                     reps=2)                  # Keep at 2 reps for speed
    
    qc = QuantumCircuit(num_features)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)
    qc.measure_all()
    
    # Use lower optimization level for faster transpilation
    qc = transpile(qc, basis_gates=['u3', 'cx'], optimization_level=1)
    
    # Create the QNN with proper output shape
    qnn = SamplerQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        interpret=parity_mapping,
        output_shape=2,
        input_gradients=True
    )
    return qnn, ansatz

def parity_mapping(bitstr):
    """More nuanced mapping for quantum measurement outcomes"""
    # Convert integer to binary string with appropriate qubit count if needed
    num_qubits = 3
    if isinstance(bitstr, int):
        bitstr = format(bitstr, f'0{num_qubits}b')
    
    # Weight the bits by position to capture more information
    # Each qubit position is weighted differently to extract more information
    weighted_sum = sum(int(bit) * (i+1) for i, bit in enumerate(bitstr))
    
    # Use modular arithmetic to map to {0,1} since output_shape=2
    return weighted_sum % 2

def preprocess_data(images, labels, debug=False):
    """Prepare data for quantum processing with better feature extraction"""
    # Flatten the images
    if len(images.shape) > 2:
        images = images.view(images.size(0), -1)
    
    # Extract improved rotation invariant features
    rot_invariant_features = extract_rotation_invariant_features(images)
    
    # Reshape labels to 1D
    labels = labels.view(-1).float()
    
    return (rot_invariant_features, labels)

def extract_rotation_invariant_features(images_tensor):
    """Extract enhanced rotation invariant features optimized for QNN processing"""
    # Convert to numpy for processing
    images_np = images_tensor.numpy()
    batch_size = images_np.shape[0]
    num_qubits = 3
    # Extract features specifically designed for histopathology
    ri_features = np.zeros((batch_size, num_qubits))
    
    for i in range(batch_size):
        # For 96x96x3 images, reshape properly before processing
        # Get the flattened image
        img = images_np[i]
        
        # For 96x96x3 images that have been flattened to 27648 elements
        if img.size == 27648:  # 96x96x3 = 27648
            # Reshape to proper dimensions for a color image
            img = img.reshape(3, 96, 96)  # [channels, height, width]
            # Convert to grayscale by averaging across color channels (axis 0)
            gray_img = np.mean(img, axis=0)  # Results in [96, 96]
        else:
            # For other cases, try to automatically determine if grayscale conversion is needed
            if len(img.shape) > 1:  # Already 2D or higher
                gray_img = np.mean(img, axis=-1) if img.shape[-1] == 3 else img
            else:  # 1D array - use a simple square reshape
                # Find the closest square for reshaping
                side = int(np.sqrt(img.size))
                gray_img = img.reshape(side, side)
        
        # Handle outliers by clipping extreme values (1% quantile on each side)
        p_low, p_high = np.percentile(gray_img, [1, 99])
        gray_img = np.clip(gray_img, p_low, p_high)
        
        # Normalize to [0,1] after outlier removal
        if p_high > p_low:  # Avoid division by zero
            gray_img = (gray_img - p_low) / (p_high - p_low)
        
        # Feature 1: Cell density with improved Gaussian smoothing
        from scipy.ndimage import gaussian_filter
        smoothed_img = gaussian_filter(gray_img, sigma=0.5)
        # Emphasize differences using nonlinear transformation
        density = 1.0 - np.mean(smoothed_img)
        ri_features[i, 0] = np.tanh(density * 2) * 0.5 + 0.5  # Scale to [0,1] with tanh
        
        # Feature 2: Textural heterogeneity using local variance
        # Use overlapping patches to capture local texture variations
        patch_size = max(2, min(gray_img.shape) // 4)
        local_vars = []
        
        for y in range(0, gray_img.shape[0] - patch_size + 1, patch_size // 2):
            for x in range(0, gray_img.shape[1] - patch_size + 1, patch_size // 2):
                patch = gray_img[y:y+patch_size, x:x+patch_size]
                local_vars.append(np.var(patch))
        
        # Use variance of local variances as a measure of heterogeneity
        if local_vars:
            texture_heterogeneity = np.var(local_vars) * 10  # Scale up for better signal
            ri_features[i, 1] = np.clip(texture_heterogeneity, 0, 1)
        
        # Feature 3: Enhanced entropy calculation using multi-scale approach
        entropies = []
        for scale in [16, 32]:  # Multiple histogram binnings for robustness
            hist, _ = np.histogram(gray_img.flatten(), bins=scale, range=(0, 1))
            hist = hist / (np.sum(hist) + 1e-8)
            entropy = -np.sum(hist * np.log2(hist + 1e-7))
            entropies.append(entropy / np.log2(scale))  # Normalize by max possible entropy
        
        # Average of multi-scale entropies
        avg_entropy = np.mean(entropies)
        ri_features[i, 2] = np.clip(avg_entropy, 0, 1)
    
    # Convert to tensor
    features_tensor = torch.tensor(ri_features, dtype=torch.float32)
    
    # Scale to [-π, π] for quantum rotation angles
    quantum_scaled_features = (features_tensor * 2 - 1) * np.pi
    
    return quantum_scaled_features

def calculate_accuracy(loader, model, device=None, criterion=None, max_batches=None, detailed=False, phase='val'):
    correct = 0
    total = 0
    loss_sum = 0.0
    all_preds = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            inputs, labels = data
            inputs, labels = preprocess_data(inputs, labels)
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Add temperature scaling for better confidence calibration
            outputs = model(inputs).squeeze() / 0.8  # Temperature < 1 for sharper predictions
            
            loss = criterion(outputs, labels)
            loss_sum += loss.item()
            
            predicted = torch.round(torch.sigmoid(outputs))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    avg_loss = loss_sum / (i + 1)
    
    # Calculate detailed metrics for research
    detailed_metrics = calculate_detailed_metrics(all_preds, all_labels)
    detailed_metrics['loss'] = avg_loss
    
    if detailed:
        print(f"\nDetailed metrics:")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"Precision: {detailed_metrics['precision']:.4f}")
        print(f"Recall: {detailed_metrics['recall']:.4f}")
        print(f"F1 Score: {detailed_metrics['f1']:.4f}")
        print(f"Confusion Matrix:")
        print(f"TN: {detailed_metrics['tn']}, FP: {detailed_metrics['fp']}")
        print(f"FN: {detailed_metrics['fn']}, TP: {detailed_metrics['tp']}")
    
    return accuracy, avg_loss, detailed_metrics

def calculate_detailed_metrics(all_preds, all_labels):
    """Calculate detailed metrics for ML research analysis"""
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score
    
    # Convert to numpy arrays if they're not already
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate basic metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0)
    
    # Calculate confusion matrix
    try:
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    except ValueError:
        # Handle case where confusion matrix doesn't have expected shape
        tn, fp, fn, tp = 0, 0, 0, 0
        
    # Calculate accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    # Try to calculate AUC if we have probability scores
    auc = None
    if hasattr(all_preds, 'shape') and len(all_preds.shape) > 1 and all_preds.shape[1] > 1:
        try:
            auc = roc_auc_score(all_labels, all_preds[:, 1])
        except:
            pass
    
    # Package all metrics
    metrics = {
        'accuracy': accuracy * 100,  # Convert to percentage
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'auc': auc
    }
    
    return metrics

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

def get_quantum_circuit_metrics(model):
    """Extract metrics from quantum circuit for analysis"""
    metrics = {
        'depth': 0,
        'num_gates': 0,
        'num_parameters': 0,
        'execution_time': 0  # This would be populated during runtime
    }
    
    try:
        if hasattr(model, 'qnn') and hasattr(model.qnn, 'circuit'):
            metrics['depth'] = model.qnn.circuit.depth()
            
            # Count gates by type
            gate_counts = {}
            for gate in model.qnn.circuit.data:
                gate_name = gate.operation.name
                gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1
            
            metrics['gate_counts'] = gate_counts
            metrics['num_gates'] = sum(gate_counts.values())
            
            # Count parameters
            if hasattr(model.qnn, 'weights'):
                metrics['num_parameters'] = len(model.qnn.weights)
            elif hasattr(model.ansatz, 'parameters'):
                metrics['num_parameters'] = len(model.ansatz.parameters)
    except Exception as e:
        print(f"Error getting quantum metrics: {e}")
    
    return metrics

def analyze_data_distribution(loader, max_samples=100):
    """Analyze the data distribution to understand potential issues"""
    label_count = {0: 0, 1: 0}
    features_stats = []
    
    print("Analyzing dataset distribution...")
    for i, (inputs, labels) in enumerate(loader):
        if i >= max_samples:
            break
            
        processed_inputs, processed_labels = preprocess_data(inputs, labels)
        
        # Count labels
        for label in processed_labels:
            label_val = int(label.item())
            label_count[label_val] = label_count.get(label_val, 0) + 1
            
        # Collect feature statistics
        features_stats.append({
            'mean': processed_inputs.mean(dim=0).numpy(),
            'std': processed_inputs.std(dim=0).numpy(),
            'min': processed_inputs.min(dim=0)[0].numpy(),
            'max': processed_inputs.max(dim=0)[0].numpy(),
        })
        
    # Print distribution analysis
    print(f"Label distribution: {label_count}")
    
    # Analyze feature statistics
    means = np.array([stats['mean'] for stats in features_stats]).mean(axis=0)
    stds = np.array([stats['std'] for stats in features_stats]).mean(axis=0)
    mins = np.array([stats['min'] for stats in features_stats]).min(axis=0)
    maxs = np.array([stats['max'] for stats in features_stats]).max(axis=0)
    
    print(f"Feature statistics:")
    for i in range(len(means)):
        print(f"Feature {i}: Mean={means[i]:.4f}, STD={stds[i]:.4f}, Range=[{mins[i]:.4f}, {maxs[i]:.4f}]")
    
    # Check if any feature has very small variance (potential issue)
    low_variance_features = [i for i, std in enumerate(stds) if std < 0.01]
    if low_variance_features:
        print(f"WARNING: Features {low_variance_features} have very low variance!")
    
    return label_count, features_stats

def test_model_output(model, loader, device=None):
    """Test if the model produces diverse outputs"""
    model.eval()
    outputs_list = []
    
    with torch.no_grad():
        for i, (inputs, _) in enumerate(loader):
            if i >= 5:  # Test with a few batches
                break
            processed_inputs, _ = preprocess_data(inputs, torch.zeros(inputs.size(0)))
            processed_inputs = processed_inputs.to(device)
            outputs = model(processed_inputs)
            outputs_list.append(outputs.cpu().numpy())
    
    all_outputs = np.concatenate(outputs_list)
    print(f"Model output statistics:")
    print(f"Mean: {all_outputs.mean():.4f}")
    print(f"STD: {all_outputs.std():.4f}")
    print(f"Min: {all_outputs.min():.4f}")
    print(f"Max: {all_outputs.max():.4f}")
    
    if all_outputs.std() < 0.01:
        print("WARNING: Model outputs have very low variance - model might be stuck!")
    
    return all_outputs

def prepare_balanced_model(trainloader, model, device=None, num_batches=1):
    """Prepare a model with balanced output distribution."""
    # Get a small sample of inputs for initialization verification
    sample_inputs = None
    for i, (inputs, _) in enumerate(trainloader):
        if i >= num_batches:
            break
        if sample_inputs is None:
            processed_inputs, _ = preprocess_data(inputs, torch.zeros(inputs.size(0)))
            sample_inputs = processed_inputs.to(device)
        else:
            more_inputs, _ = preprocess_data(inputs, torch.zeros(inputs.size(0)))
            sample_inputs = torch.cat([sample_inputs, more_inputs.to(device)], dim=0)
    
    # Verify and correct initialization if needed
    print("\n=== VERIFYING MODEL INITIALIZATION ===")
    model.verify_output_distribution(sample_inputs, target_mean=0.0)
    
    return model
