import tensorflow as tf
import numpy as np
import os
import json
import trimesh
import scipy.io
import h5py
from PIL import Image
from pathlib import Path
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

class ShapeNetCoreV2Dataset:
    """Dataset loader for ShapeNetCore.v2"""
    
    # ShapeNetCore.v2 category IDs and names
    CATEGORY_MAP = {
        '02691156': 'airplane',
        '02828884': 'bench',
        '02933112': 'cabinet',
        '02958343': 'car',
        '03001627': 'chair',
        '03211117': 'display',
        '03636649': 'lamp',
        '03691459': 'speaker',
        '04090263': 'rifle',
        '04256520': 'sofa',
        '04379243': 'table',
        '04401088': 'telephone',
        '04530566': 'watercraft'
    }

    def __init__(self, 
                 dataset_path,
                 image_size=(256, 256),
                 num_points=1024,
                 categories=None,
                 splits_file=None):
        """
        Initialize ShapeNetCore.v2 dataset loader
        
        Args:
            dataset_path: Path to ShapeNetCore.v2 root directory
            image_size: Target image size for rendered images
            num_points: Number of points to sample from each mesh
            categories: List of category IDs to use (None for all)
            splits_file: Path to train/test splits file (None for default)
        """
        self.dataset_path = Path(dataset_path)
        self.image_size = image_size
        self.num_points = num_points
        self.categories = categories or list(self.CATEGORY_MAP.keys())
        
        # Load official train/test splits
        self.splits_file = splits_file or self.dataset_path / 'train_test_split.json'
        with open(self.splits_file, 'r') as f:
            self.splits = json.load(f)
            
        # Get all valid model paths
        self.models = self._get_model_paths()
        
        print(f"Loaded {len(self.models)} models from {len(self.categories)} categories")
        
    def _get_model_paths(self):
        """Get paths for all valid models"""
        models = []
        
        for category in self.categories:
            category_path = self.dataset_path / category
            if not category_path.is_dir():
                continue
                
            # Get all models in category
            for model_id in os.listdir(category_path):
                model_path = category_path / model_id
                
                # Check if model has required files
                if not self._validate_model(model_path):
                    continue
                    
                # Get split (train/test) for this model
                split = self.splits.get(model_id, 'train')
                
                models.append({
                    'category': category,
                    'category_name': self.CATEGORY_MAP[category],
                    'model_id': model_id,
                    'path': model_path,
                    'split': split
                })
        
        return models
    
    def _validate_model(self, model_path):
        """Check if model has all required files"""
        required_files = [
            'models/model_normalized.obj',  # Normalized mesh
            'models/model_normalized.mtl',  # Material file
            'rendering/rendering_metadata.txt'  # Rendering metadata
        ]
        
        return all((model_path / f).exists() for f in required_files)
    
    def _load_point_cloud(self, model_path, augment=False):
        """Load and sample point cloud from model"""
        mesh_path = model_path / 'models/model_normalized.obj'
        
        try:
            mesh = trimesh.load(str(mesh_path), force='mesh')
            
            # Sample points from mesh surface
            points = mesh.sample(self.num_points)
            
            # Center and normalize to unit cube
            center = points.mean(axis=0)
            points = points - center
            scale = np.abs(points).max()
            points = points / scale
            
            if augment:
                points = self._augment_point_cloud(points)
            
            return points.astype(np.float32)
            
        except Exception as e:
            print(f"Error loading mesh {mesh_path}: {e}")
            return None
    
    def _load_renderings(self, model_path):
        """Load rendered images and their metadata"""
        rendering_path = model_path / 'rendering'
        metadata_path = rendering_path / 'rendering_metadata.txt'
        
        # Load rendering metadata
        with open(metadata_path, 'r') as f:
            metadata = [line.strip().split(' ') for line in f.readlines()]
        
        images = []
        cameras = []
        
        for i, (azimuth, elevation, distance, _) in enumerate(metadata):
            img_path = rendering_path / f'{i:02d}.png'
            if not img_path.exists():
                continue
            
            # Load and preprocess image
            try:
                img = load_img(str(img_path), target_size=self.image_size)
                img = img_to_array(img)
                images.append(img)
                
                # Convert camera parameters
                azimuth = float(azimuth)
                elevation = float(elevation)
                distance = float(distance)
                
                # Create camera matrix
                camera_matrix = self._create_camera_matrix(azimuth, elevation, distance)
                cameras.append(camera_matrix)
                
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue
        
        if not images:
            return None, None
            
        return np.array(images, dtype=np.float32) / 255.0, np.array(cameras, dtype=np.float32)
    
    def _create_camera_matrix(self, azimuth, elevation, distance):
        """Create 4x4 camera matrix from rendering parameters"""
        # Convert angles to radians
        azimuth = np.radians(azimuth)
        elevation = np.radians(elevation)
        
        # Create rotation matrix
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(elevation), -np.sin(elevation)],
            [0, np.sin(elevation), np.cos(elevation)]
        ])
        
        Ry = np.array([
            [np.cos(azimuth), 0, np.sin(azimuth)],
            [0, 1, 0],
            [-np.sin(azimuth), 0, np.cos(azimuth)]
        ])
        
        R = Rx @ Ry
        
        # Create translation vector
        t = np.array([0, 0, distance])
        
        # Create 4x4 transformation matrix
        matrix = np.eye(4)
        matrix[:3, :3] = R
        matrix[:3, 3] = t
        
        return matrix
    
    def _augment_point_cloud(self, points):
        """Apply augmentation to point cloud"""
        # Random rotation around y-axis
        theta = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array([
            [np.cos(theta), 0, -np.sin(theta)],
            [0, 1, 0],
            [np.sin(theta), 0, np.cos(theta)]
        ])
        points = np.matmul(points, rotation_matrix)
        
        # Random jitter
        jitter = np.random.normal(0, 0.01, points.shape)
        points += jitter
        
        # Random scaling
        scale = np.random.uniform(0.8, 1.2)
        points *= scale
        
        return points
    
    def _augment_image(self, image):
        """Apply augmentation to image"""
        # Color jittering
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        image = tf.image.random_saturation(image, 0.8, 1.2)
        
        # Random horizontal flip
        image = tf.image.random_flip_left_right(image)
        
        return tf.clip_by_value(image, 0, 1)

class ShapeNetDataGenerator(tf.keras.utils.Sequence):
    """Data generator for ShapeNetCore.v2"""
    
    def __init__(self,
                 dataset,
                 split='train',
                 batch_size=32,
                 shuffle=True,
                 augment=True):
        """
        Initialize data generator
        
        Args:
            dataset: ShapeNetCoreV2Dataset instance
            split: 'train' or 'test'
            batch_size: Batch size
            shuffle: Whether to shuffle data each epoch
            augment: Whether to apply augmentation
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        
        # Filter models by split
        self.models = [m for m in dataset.models if m['split'] == split]
        self.indices = np.arange(len(self.models))
        
        self.on_epoch_end()
    
    def __len__(self):
        """Number of batches per epoch"""
        return len(self.indices) // self.batch_size
    
    def __getitem__(self, idx):
        """Get batch at position idx"""
        start_idx = idx * self.batch_size
        batch_indices = self.indices[start_idx:start_idx + self.batch_size]
        
        # Initialize batch arrays
        batch_images = np.zeros((self.batch_size, *self.dataset.image_size, 3))
        batch_points = np.zeros((self.batch_size, self.dataset.num_points, 3))
        batch_cameras = np.zeros((self.batch_size, 4, 4))
        
        valid_samples = 0
        
        for i, idx in enumerate(batch_indices):
            model = self.models[idx]
            
            # Load point cloud
            points = self.dataset._load_point_cloud(
                model['path'],
                augment=self.augment
            )
            
            if points is None:
                continue
                
            # Load renderings
            images, cameras = self.dataset._load_renderings(model['path'])
            
            if images is None:
                continue
                
            # Randomly select one view
            view_idx = np.random.randint(len(images))
            image = images[view_idx]
            camera = cameras[view_idx]
            
            # Apply image augmentation if needed
            if self.augment:
                image = self.dataset._augment_image(image)
            
            batch_images[valid_samples] = image
            batch_points[valid_samples] = points
            batch_cameras[valid_samples] = camera
            
            valid_samples += 1
            
            if valid_samples == self.batch_size:
                break
        
        if valid_samples == 0:
            return self.__getitem__((idx + 1) % len(self))
        
        # Trim batch to valid samples
        return {
            'images': batch_images[:valid_samples],
            'points': batch_points[:valid_samples],
            'cameras': batch_cameras[:valid_samples]
        }
    
    def on_epoch_end(self):
        """Called at the end of every epoch"""
        if self.shuffle:
            np.random.shuffle(self.indices)

def prepare_shapenet_dataset(dataset_path, batch_size=32):
    """
    Prepare ShapeNetCore.v2 dataset for training
    
    Args:
        dataset_path: Path to ShapeNetCore.v2 dataset
        batch_size: Batch size for training
    
    Returns:
        train_generator, test_generator
    """
    # Initialize dataset
    dataset = ShapeNetCoreV2Dataset(dataset_path)
    
    # Create generators
    train_generator = ShapeNetDataGenerator(
        dataset,
        split='train',
        batch_size=batch_size,
        shuffle=True,
        augment=True
    )
    
    test_generator = ShapeNetDataGenerator(
        dataset,
        split='test',
        batch_size=batch_size,
        shuffle=False,
        augment=False
    )
    
    return train_generator, test_generator

# Modified training function to work with new data format
def train_reconstruction_model(dataset_path, output_dir, epochs=100, batch_size=32):
    """
    Train the 3D reconstruction model using ShapeNetCore.v2
    
    Args:
        dataset_path: Path to ShapeNetCore.v2 dataset
        output_dir: Directory to save model and results
        epochs: Number of training epochs
        batch_size: Batch size for training
    """
    # Prepare dataset
    train_generator, test_generator = prepare_shapenet_dataset(
        dataset_path,
        batch_size=batch_size
    )
    
    # Build model
    model = build_complete_model()
    
    # Define optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    
    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Training phase
        train_losses = []
        for batch_idx in range(len(train_generator)):
            batch_data = train_generator[batch_idx]
            
            with tf.GradientTape() as tape:
                # Forward pass
                point_cloud_pred, projected_points, depth = model(batch_data['images'])
                
                # Calculate losses
                pc_loss = point_cloud_loss(batch_data['points'], point_cloud_pred)
                depth_loss = depth_consistency_loss(depth, projected_points)
                total_loss = pc_loss + 0.1 * depth_loss
                
            # Backpropagation
            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            train_losses.append(total_loss.numpy())
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}, Loss: {total_loss:.4f}")
        
        avg_train_loss = np.mean(train_losses)
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        
        # Testing phase
        test_losses = []
        for batch_idx in range(len(test_generator)):
            batch_data = test_generator[batch_idx]
            point_cloud_pred, projected_points, depth = model(batch_data['images'])
            
            pc_loss = point_cloud_loss(batch_data['points'], point_cloud_pred)
            depth_loss = depth_consistency_loss(depth, projected_points)
            total_loss = pc_loss + 0.1 * depth_loss
            test_losses.append(total_loss.numpy())
        
        avg_test_loss = np.mean(test_losses)
        print(f"Average Test Loss: {avg_test_loss:.4f}")

        # Save model checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(output_dir, f'model_epoch_{epoch+1}')
            model.save_weights(checkpoint_path)
            
            # Save training metrics
            metrics = {
                'epoch': epoch + 1,
                'train_loss': float(avg_train_loss),
                'test_loss': float(avg_test_loss),
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            with open(os.path.join(output_dir, f'metrics_epoch_{epoch+1}.json'), 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Generate and save validation visualizations
            visualize_results(
                model,
                test_generator,
                output_dir,
                epoch + 1,
                num_samples=5
            )

def visualize_results(model, test_generator, output_dir, epoch, num_samples=5):
    """
    Generate and save visualization of reconstruction results
    
    Args:
        model: Trained model
        test_generator: Test data generator
        output_dir: Output directory
        epoch: Current epoch number
        num_samples: Number of samples to visualize
    """
    # Create visualization directory
    vis_dir = os.path.join(output_dir, f'visualizations_epoch_{epoch}')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Get random batch
    batch_data = test_generator[np.random.randint(len(test_generator))]
    
    for i in range(min(num_samples, len(batch_data['images']))):
        # Get single sample
        image = batch_data['images'][i]
        gt_points = batch_data['points'][i]
        
        # Generate prediction
        point_cloud_pred, projected_points, depth = model(tf.expand_dims(image, 0))
        pred_points = point_cloud_pred[0]
        
        # Create visualization
        fig = plt.figure(figsize=(15, 5))
        
        # Plot input image
        ax1 = fig.add_subplot(131)
        ax1.imshow(image)
        ax1.set_title('Input Image')
        ax1.axis('off')
        
        # Plot ground truth point cloud
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.scatter(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2], c='b', s=1)
        ax2.set_title('Ground Truth')
        ax2.axis('off')
        
        # Plot predicted point cloud
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], c='r', s=1)
        ax3.set_title('Prediction')
        ax3.axis('off')
        
        # Save figure
        plt.savefig(os.path.join(vis_dir, f'sample_{i}.png'))
        plt.close()

def evaluate_model(model, test_generator, output_dir):
    """
    Evaluate trained model on test set
    
    Args:
        model: Trained model
        test_generator: Test data generator
        output_dir: Output directory
    """
    all_chamfer_distances = []
    all_depth_errors = []
    category_metrics = defaultdict(list)
    
    for batch_idx in range(len(test_generator)):
        batch_data = test_generator[batch_idx]
        point_cloud_pred, projected_points, depth = model(batch_data['images'])
        
        # Calculate metrics
        for i in range(len(batch_data['images'])):
            # Chamfer distance
            cd = chamfer_distance(
                batch_data['points'][i:i+1],
                point_cloud_pred[i:i+1]
            )
            all_chamfer_distances.append(cd)
            
            # Depth error
            depth_error = depth_consistency_loss(
                depth[i:i+1],
                projected_points[i:i+1]
            )
            all_depth_errors.append(depth_error)
            
            # Store by category
            category = batch_data['categories'][i]
            category_metrics[category].append({
                'chamfer_distance': cd,
                'depth_error': depth_error
            })
    
    # Calculate overall metrics
    overall_metrics = {
        'mean_chamfer_distance': float(np.mean(all_chamfer_distances)),
        'std_chamfer_distance': float(np.std(all_chamfer_distances)),
        'mean_depth_error': float(np.mean(all_depth_errors)),
        'std_depth_error': float(np.std(all_depth_errors))
    }
    
    # Calculate per-category metrics
    category_results = {}
    for category, metrics in category_metrics.items():
        cd_values = [m['chamfer_distance'] for m in metrics]
        depth_values = [m['depth_error'] for m in metrics]
        
        category_results[category] = {
            'mean_chamfer_distance': float(np.mean(cd_values)),
            'std_chamfer_distance': float(np.std(cd_values)),
            'mean_depth_error': float(np.mean(depth_values)),
            'std_depth_error': float(np.std(depth_values)),
            'num_samples': len(metrics)
        }
    
    # Save evaluation results
    results = {
        'overall_metrics': overall_metrics,
        'category_results': category_results,
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train 3D reconstruction model on ShapeNetCore.v2')
    parser.add_argument('--dataset_path', required=True, help='Path to ShapeNetCore.v2 dataset')
    parser.add_argument('--output_dir', required=True, help='Output directory for models and results')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation after training')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train model
    model = train_reconstruction_model(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Run evaluation if requested
    if args.evaluate:
        _, test_generator = prepare_shapenet_dataset(
            args.dataset_path,
            batch_size=args.batch_size
        )
        
        evaluation_results = evaluate_model(
            model,
            test_generator,
            args.output_dir
        )
        
        print("Evaluation Results:")
        print(json.dumps(evaluation_results['overall_metrics'], indent=2))