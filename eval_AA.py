import torch
import torch.nn as nn
from autoattack import AutoAttack
import torchvision
import torchvision.transforms as transforms
import csv
import os
from models import *

# Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epsilon = 0.0314  # 8/255
num_classes = 10

# Adjust your path for the checkpoints
checkpoint_base = './models'
epoch_range = range(74, 85)  
checkpoint_paths = [
    f'{checkpoint_base}/epoch_{epoch}_model.pth' 
    for epoch in epoch_range
]

# Output directory for results
output_dir = 'AutoAttack_Results_Epochs'
os.makedirs(output_dir, exist_ok=True)

# Prepare test dataset
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

test_dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=False, 
    download=True, 
    transform=transform_test
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=100, 
    shuffle=False, 
    num_workers=4
)

def evaluate_autoattack(checkpoint_path, epoch_num):
    """
    Evaluate a single checkpoint using AutoAttack
    
    Args:
        checkpoint_path: Path to the model checkpoint
        epoch_num: Epoch number for naming
    """
    print(f"\n{'='*60}")
    print(f"Epoch {epoch_num}: Running AutoAttack on {checkpoint_path}")
    print(f"{'='*60}\n")
    
    # Initialize model
    net = ResNet18()
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return None
    
    try:
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        net.eval()
        print(f"✓ Loaded checkpoint successfully")
    except Exception as e:
        print(f"ERROR loading checkpoint: {e}")
        return None
    
    # Prepare data for AutoAttack
    print("Loading test data...")
    x_test = []
    y_test = []
    
    for x, y in test_loader:
        x_test.append(x)
        y_test.append(y)
    
    x_test = torch.cat(x_test, dim=0).to(device)
    y_test = torch.cat(y_test, dim=0).to(device)
    
    print(f"Test data shape: {x_test.shape}")
    
    # Initialize AutoAttack
    adversary = AutoAttack(
        net, 
        norm='Linf', 
        eps=epsilon, 
        version='standard', 
        verbose=True
    )
    
    # Run AutoAttack
    print("\nStarting AutoAttack...")
    try:
        x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=250)
    except Exception as e:
        print(f"ERROR during AutoAttack: {e}")
        return None
    
    # Evaluate per-class robust accuracy
    print("\nCalculating per-class accuracies...")
    with torch.no_grad():
        logits_adv = net(x_adv)
        _, predicted = logits_adv.max(1)
        
        rob_correct = [0] * num_classes
        class_totals = [0] * num_classes
        
        for class_id in range(num_classes):
            class_mask = (y_test == class_id)
            class_totals[class_id] = class_mask.sum().item()
            rob_correct[class_id] = (predicted.eq(y_test) & class_mask).sum().item()
        
        rob_correct_perc = [
            (rob / total * 100) if total > 0 else 0 
            for rob, total in zip(rob_correct, class_totals)
        ]
        
        overall_accuracy = (predicted.eq(y_test).sum().item() / len(y_test)) * 100
    
    # Save results to CSV
    csv_path = f'{output_dir}/AutoAttack_epoch_{epoch_num}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Checkpoint', checkpoint_path])
        writer.writerow(['Epoch', epoch_num])
        writer.writerow(['Epsilon', epsilon])
        writer.writerow([])
        writer.writerow(['Class', 'Robust_Accuracy (%)'])
        
        for i, acc in enumerate(rob_correct_perc):
            writer.writerow([f'Class_{i}', f'{acc:.2f}'])
        
        writer.writerow([])
        writer.writerow(['Overall_Accuracy', f'{overall_accuracy:.2f}'])
        writer.writerow(['Average_Per_Class', f'{sum(rob_correct_perc)/len(rob_correct_perc):.2f}'])
        writer.writerow(['Min_Class_Accuracy', f'{min(rob_correct_perc):.2f}'])
        writer.writerow(['Max_Class_Accuracy', f'{max(rob_correct_perc):.2f}'])
    
    # Print results
    print(f"\n{'='*60}")
    print(f"AutoAttack Results - Epoch {epoch_num}:")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    print(f"\nPer-Class Accuracies:")
    for i, acc in enumerate(rob_correct_perc):
        print(f"  Class {i}: {acc:.2f}%")
    print(f"\nAverage: {sum(rob_correct_perc)/len(rob_correct_perc):.2f}%")
    print(f"Min: {min(rob_correct_perc):.2f}%")
    print(f"Max: {max(rob_correct_perc):.2f}%")
    print(f"{'='*60}")
    print(f"Results saved to: {csv_path}\n")
    
    return {
        'epoch': epoch_num,
        'overall': overall_accuracy,
        'accuracies': rob_correct_perc,
        'average': sum(rob_correct_perc) / len(rob_correct_perc),
        'min': min(rob_correct_perc),
        'max': max(rob_correct_perc)
    }

def main():
    print("="*60)
    print("AutoAttack Evaluation Script - Epochs 101-110")
    print("="*60)
    print(f"Device: {device}")
    print(f"Epsilon: {epsilon} (8/255)")
    print(f"Checkpoint directory: {checkpoint_base}")
    print(f"Epoch range: {min(epoch_range)} to {max(epoch_range)}")
    print(f"Total checkpoints: {len(checkpoint_paths)}")
    print("="*60)
    
    results_summary = []
    
    for i, (epoch_num, checkpoint_path) in enumerate(zip(epoch_range, checkpoint_paths), 1):
        print(f"\n\n{'#'*60}")
        print(f"Progress: {i}/{len(checkpoint_paths)} - Epoch {epoch_num}")
        print(f"{'#'*60}")
        
        result = evaluate_autoattack(checkpoint_path, epoch_num)
        
        if result is not None:
            results_summary.append(result)
    
    # Save detailed summary
    summary_path = f'{output_dir}/AutoAttack_Summary_All_Epochs.csv'
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Overall_Accuracy', 'Average_Per_Class', 'Min_Accuracy', 'Max_Accuracy'] + 
                       [f'Class_{i}' for i in range(num_classes)])
        
        for result in results_summary:
            row = [
                result['epoch'],
                f"{result['overall']:.2f}",
                f"{result['average']:.2f}",
                f"{result['min']:.2f}",
                f"{result['max']:.2f}"
            ] + [f"{acc:.2f}" for acc in result['accuracies']]
            writer.writerow(row)
    
    # Save compact summary (just key metrics)
    compact_path = f'{output_dir}/AutoAttack_Summary_Compact.csv'
    with open(compact_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Overall_Accuracy', 'Min_Class_Accuracy'])
        
        for result in results_summary:
            writer.writerow([
                result['epoch'],
                f"{result['overall']:.2f}",
                f"{result['min']:.2f}"
            ])
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY OF ALL EPOCHS")
    print("="*60)
    print(f"{'Epoch':<8} {'Overall':<12} {'Avg/Class':<12} {'Min':<8} {'Max':<8}")
    print("-"*60)
    for result in results_summary:
        print(f"{result['epoch']:<8} {result['overall']:>6.2f}%     "
              f"{result['average']:>6.2f}%     {result['min']:>6.2f}% {result['max']:>6.2f}%")
    
    if results_summary:
        best_overall = max(results_summary, key=lambda x: x['overall'])
        best_min = max(results_summary, key=lambda x: x['min'])
        
        print("\n" + "="*60)
        print(f"Best Overall Accuracy: Epoch {best_overall['epoch']} ({best_overall['overall']:.2f}%)")
        print(f"Best Min Class Accuracy: Epoch {best_min['epoch']} ({best_min['min']:.2f}%)")
        print("="*60)
    
    print(f"\nDetailed summary saved to: {summary_path}")
    print(f"Compact summary saved to: {compact_path}")
    print(f"Individual results in: {output_dir}/")
    print("="*60)

if __name__ == '__main__':
    main()import torch
import torch.nn as nn
from autoattack import AutoAttack
import torchvision
import torchvision.transforms as transforms
import csv
import os
from models import *

# Configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epsilon = 0.0314  # 8/255
num_classes = 10

# Generate checkpoint paths for epochs 101-110
checkpoint_base = './models'
epoch_range = range(101, 111)  # 101 to 110 inclusive
checkpoint_paths = [
    f'{checkpoint_base}/epoch_{epoch}_model.pth' 
    for epoch in epoch_range
]

# Output directory for results
output_dir = 'AutoAttack_Results_Epochs'
os.makedirs(output_dir, exist_ok=True)

# Prepare test dataset
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

test_dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=False, 
    download=True, 
    transform=transform_test
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=100, 
    shuffle=False, 
    num_workers=4
)

def evaluate_autoattack(checkpoint_path, epoch_num):
    """
    Evaluate a single checkpoint using AutoAttack
    
    Args:
        checkpoint_path: Path to the model checkpoint
        epoch_num: Epoch number for naming
    """
    print(f"\n{'='*60}")
    print(f"Epoch {epoch_num}: Running AutoAttack on {checkpoint_path}")
    print(f"{'='*60}\n")
    
    # Initialize model
    net = ResNet18()
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return None
    
    try:
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        net.eval()
        print(f"✓ Loaded checkpoint successfully")
    except Exception as e:
        print(f"ERROR loading checkpoint: {e}")
        return None
    
    # Prepare data for AutoAttack
    print("Loading test data...")
    x_test = []
    y_test = []
    
    for x, y in test_loader:
        x_test.append(x)
        y_test.append(y)
    
    x_test = torch.cat(x_test, dim=0).to(device)
    y_test = torch.cat(y_test, dim=0).to(device)
    
    print(f"Test data shape: {x_test.shape}")
    
    # Initialize AutoAttack
    adversary = AutoAttack(
        net, 
        norm='Linf', 
        eps=epsilon, 
        version='standard', 
        verbose=True
    )
    
    # Run AutoAttack
    print("\nStarting AutoAttack...")
    try:
        x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=250)
    except Exception as e:
        print(f"ERROR during AutoAttack: {e}")
        return None
    
    # Evaluate per-class robust accuracy
    print("\nCalculating per-class accuracies...")
    with torch.no_grad():
        logits_adv = net(x_adv)
        _, predicted = logits_adv.max(1)
        
        rob_correct = [0] * num_classes
        class_totals = [0] * num_classes
        
        for class_id in range(num_classes):
            class_mask = (y_test == class_id)
            class_totals[class_id] = class_mask.sum().item()
            rob_correct[class_id] = (predicted.eq(y_test) & class_mask).sum().item()
        
        rob_correct_perc = [
            (rob / total * 100) if total > 0 else 0 
            for rob, total in zip(rob_correct, class_totals)
        ]
        
        overall_accuracy = (predicted.eq(y_test).sum().item() / len(y_test)) * 100
    
    # Save results to CSV
    csv_path = f'{output_dir}/AutoAttack_epoch_{epoch_num}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Checkpoint', checkpoint_path])
        writer.writerow(['Epoch', epoch_num])
        writer.writerow(['Epsilon', epsilon])
        writer.writerow([])
        writer.writerow(['Class', 'Robust_Accuracy (%)'])
        
        for i, acc in enumerate(rob_correct_perc):
            writer.writerow([f'Class_{i}', f'{acc:.2f}'])
        
        writer.writerow([])
        writer.writerow(['Overall_Accuracy', f'{overall_accuracy:.2f}'])
        writer.writerow(['Average_Per_Class', f'{sum(rob_correct_perc)/len(rob_correct_perc):.2f}'])
        writer.writerow(['Min_Class_Accuracy', f'{min(rob_correct_perc):.2f}'])
        writer.writerow(['Max_Class_Accuracy', f'{max(rob_correct_perc):.2f}'])
    
    # Print results
    print(f"\n{'='*60}")
    print(f"AutoAttack Results - Epoch {epoch_num}:")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    print(f"\nPer-Class Accuracies:")
    for i, acc in enumerate(rob_correct_perc):
        print(f"  Class {i}: {acc:.2f}%")
    print(f"\nAverage: {sum(rob_correct_perc)/len(rob_correct_perc):.2f}%")
    print(f"Min: {min(rob_correct_perc):.2f}%")
    print(f"Max: {max(rob_correct_perc):.2f}%")
    print(f"{'='*60}")
    print(f"Results saved to: {csv_path}\n")
    
    return {
        'epoch': epoch_num,
        'overall': overall_accuracy,
        'accuracies': rob_correct_perc,
        'average': sum(rob_correct_perc) / len(rob_correct_perc),
        'min': min(rob_correct_perc),
        'max': max(rob_correct_perc)
    }

def main():
    print("="*60)
    print("AutoAttack Evaluation Script - Epochs 101-110")
    print("="*60)
    print(f"Device: {device}")
    print(f"Epsilon: {epsilon} (8/255)")
    print(f"Checkpoint directory: {checkpoint_base}")
    print(f"Epoch range: {min(epoch_range)} to {max(epoch_range)}")
    print(f"Total checkpoints: {len(checkpoint_paths)}")
    print("="*60)
    
    results_summary = []
    
    for i, (epoch_num, checkpoint_path) in enumerate(zip(epoch_range, checkpoint_paths), 1):
        print(f"\n\n{'#'*60}")
        print(f"Progress: {i}/{len(checkpoint_paths)} - Epoch {epoch_num}")
        print(f"{'#'*60}")
        
        result = evaluate_autoattack(checkpoint_path, epoch_num)
        
        if result is not None:
            results_summary.append(result)
    
    # Save detailed summary
    summary_path = f'{output_dir}/AutoAttack_Summary_All_Epochs.csv'
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Overall_Accuracy', 'Average_Per_Class', 'Min_Accuracy', 'Max_Accuracy'] + 
                       [f'Class_{i}' for i in range(num_classes)])
        
        for result in results_summary:
            row = [
                result['epoch'],
                f"{result['overall']:.2f}",
                f"{result['average']:.2f}",
                f"{result['min']:.2f}",
                f"{result['max']:.2f}"
            ] + [f"{acc:.2f}" for acc in result['accuracies']]
            writer.writerow(row)
    
    # Save compact summary (just key metrics)
    compact_path = f'{output_dir}/AutoAttack_Summary_Compact.csv'
    with open(compact_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Overall_Accuracy', 'Min_Class_Accuracy'])
        
        for result in results_summary:
            writer.writerow([
                result['epoch'],
                f"{result['overall']:.2f}",
                f"{result['min']:.2f}"
            ])
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY OF ALL EPOCHS")
    print("="*60)
    print(f"{'Epoch':<8} {'Overall':<12} {'Avg/Class':<12} {'Min':<8} {'Max':<8}")
    print("-"*60)
    for result in results_summary:
        print(f"{result['epoch']:<8} {result['overall']:>6.2f}%     "
              f"{result['average']:>6.2f}%     {result['min']:>6.2f}% {result['max']:>6.2f}%")
    
    if results_summary:
        best_overall = max(results_summary, key=lambda x: x['overall'])
        best_min = max(results_summary, key=lambda x: x['min'])
        
        print("\n" + "="*60)
        print(f"Best Overall Accuracy: Epoch {best_overall['epoch']} ({best_overall['overall']:.2f}%)")
        print(f"Best Min Class Accuracy: Epoch {best_min['epoch']} ({best_min['min']:.2f}%)")
        print("="*60)
    
    print(f"\nDetailed summary saved to: {summary_path}")
    print(f"Compact summary saved to: {compact_path}")
    print(f"Individual results in: {output_dir}/")
    print("="*60)

if __name__ == '__main__':
    main()