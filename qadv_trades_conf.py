import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from autoattack import AutoAttack
import torchvision
import torchvision.transforms as transforms
import sys
from pathlib import Path
import os
import csv
import time
from itertools import product

from models import *

if len(sys.argv) > 1:
    file_specific_name = sys.argv[1]
else:
    file_specific_name = "default"


destination_epoch = 110
warmup_epochs = 70
num_classes = 10
incr = 1e-8
beta = 6
epsilon = 0.0314
k = 7
alpha = 0.00784
target_mode = "Targeted" # Meaningless Placeholder


# Please specify your parameters down
# if you want only 1 run, have only single parameter in each list


model_types = ["Mult"]
target_modes = ["Untargeted"]
lrs = [1e-1]
lambdas = [0.1]


device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)

class LinfPGDAttack(object):
    def __init__(self, model):
        self.model = model
        self.confusion_matrix = torch.zeros(num_classes, num_classes, device=device)

    def random_targets(self, y):
        '''
        Moves the label class by 1 to 9 points.
        Guarantess that y != y_t
        '''
        num_classes = len(train_dataset.classes)
        return (y + torch.randint(1, num_classes, (y.size(0),), device=y.device)) % num_classes


    def perturb(self, x_natural, y):
        x = x_natural.detach().clone()
        # x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
        x += 0.001 * torch.randn(x.shape).cuda().detach()
        for i in range(k):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - epsilon), x_natural + epsilon)
            x = torch.clamp(x, 0, 1)
        return x

    def perturb_target(self, x_natural, y):
        x = x_natural.detach().clone()
        # x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
        x += 0.001 * torch.randn(x.shape).cuda().detach()
        y_t = self.random_targets(y)
        for i in range(k):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, y_t)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() - alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - epsilon), x_natural + epsilon)
            x = torch.clamp(x, 0, 1)
        return x

    def perturb_confusion(self, x_natural, y):
        x = x_natural.detach().clone()
        # x = x + torch.zeros_like(x).uniform_(-epsilon, epsilon)
        x += 0.001 * torch.randn(x.shape).cuda().detach()
        
        # Sample targets based on confusion matrix
        y_t = torch.zeros_like(y)
        for i in range(len(y)):
            confusion_probs = self.confusion_matrix[y[i]].clone()
            confusion_probs[y[i]] = 0  # Don't target the true class
            if confusion_probs.sum() < 1e-8:  # Cold start: use random
                y_t[i] = self.random_targets(y[i:i+1])[0]
            else:
                y_t[i] = torch.multinomial(confusion_probs, 1)
        
        for i in range(k):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, y_t)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() - alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - epsilon), x_natural + epsilon)
            x = torch.clamp(x, 0, 1)
        return x

    def perturb_test(self, x, y):
        self.model.eval()
        x_adv = x.detach().clone()
        x_adv += 0.001 * torch.randn(x.shape).cuda().detach()
        for i in range(k):
            x_adv.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x_adv)
                loss = F.cross_entropy(logits, y)
            mask = torch.ones(len(y)).cuda()
            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() +alpha * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
            x_adv = torch.clamp(x_adv, 0, 1)
            x_adv = x_adv.detach()
        return x_adv


net = ResNet18()
net = net.to(device)
net = torch.nn.DataParallel(net)
cudnn.benchmark = True

adversary = LinfPGDAttack(net)
optimizer = optim.SGD(net.parameters(), lr=0, momentum=0.9, weight_decay=0.0002)
#just a placeholder for optimizer learning rate


Q = torch.zeros(num_classes, device=device)
q_lr = 0.05
criterion = torch.nn.CrossEntropyLoss()

# Global variables to track previous epoch metrics
prev_max_loss = torch.tensor(float('inf'), device=device)
prev_avg_loss = torch.tensor(float('inf'), device=device)

def train(epoch, lambda_rate, model_type):
    global prev_max_loss, prev_avg_loss
    
    print('\n[ Train epoch: %d ]' % epoch)
    net.train()
    
    epoch_class_loss_sum = torch.zeros(num_classes, device=device)
    epoch_class_count = torch.zeros(num_classes, device=device) 
    train_loss = 0.0
    correct = 0
    total = 0
    rob_correct = [0]*num_classes
    loss_model_sum = 0.0
    loss_power_sum = 0.0
    total = 0

    class_total = torch.zeros(num_classes, device=device)

    for batch_idx, (x, y) in enumerate(train_loader):

        # creating adversarial images
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        net.eval()
        if epoch > 20:
            x_adv = adversary.perturb_confusion(x, y)
        else:
            x_adv = adversary.perturb(x, y)
        net.train()
        x_hat = net(x_adv)
        logits_nat = net(x)
        #-----------------------------------------------------------------------------------------------------------
        # Confusion Matrix Update
        with torch.no_grad():
            _, pred_adv = x_hat.max(1)
            for i in range(len(y)):
                true_label, pred_label = y[i].item(), pred_adv[i].item()
                adversary.confusion_matrix[true_label, pred_label] = \
                    0.99 * adversary.confusion_matrix[true_label, pred_label] + 0.01
            # Normalize rows
            row_sums = adversary.confusion_matrix.sum(dim=1, keepdim=True).clamp(min=1e-8)
            adversary.confusion_matrix = adversary.confusion_matrix / row_sums
        #-----------------------------------------------------------------------------------------------------------
        if epoch <= warmup_epochs:
            loss_natural = torch.nn.CrossEntropyLoss()(logits_nat, y)
            loss_robust = nn.KLDivLoss(reduction='sum')(F.log_softmax(x_hat + incr, dim=1), F.softmax(logits_nat, dim=1)+incr) / (len(x) + incr)
            loss = loss_natural + beta * loss_robust
        #-----------------------------------------------------------------------------------------------------------
        else:
            probs = F.softmax(x_hat, dim=1)
            class_ids = torch.multinomial(probs, 1).squeeze(1)
            Reward = ((class_ids == y).float() * 2 - 1)
            with torch.no_grad():
                for c in range(num_classes):
                    mask = (class_ids == c)
                    if mask.any():
                        raw_Q = (1 - q_lr) * Q[c] + q_lr * Reward[mask].mean()
                        Q[c] = torch.clamp(raw_Q, min=-0.5, max=1.0) # avoiding low expectations
            Advantage = Reward - Q[class_ids]
            Advantage = Advantage.clamp(-2.0, 2.0)

            p_at_st = probs[range(x.size(0)), class_ids] 
            loss_model = -1 * (torch.log(p_at_st) * Advantage).mean() 
            #-----------------------------------------------------------------------------------------------------------
            # Power-based Loss accumulate perclass losses across batches
            with torch.no_grad():
                for c in range(num_classes):
                    mask = (y == c)
                    if mask.any():
                        class_total[c] += mask.sum()
                        epoch_class_loss_sum[c] += F.cross_entropy(x_hat[mask], y[mask]) * mask.sum()
                        epoch_class_count[c] += mask.sum()

            # Calculate batch loss_power for this batch only 
            per_class_loss_batch = torch.zeros(num_classes, device=device)
            for c in range(num_classes):
                mask = (y == c)
                if mask.any():
                    per_class_loss_batch[c] = F.cross_entropy(x_hat[mask], y[mask])
            
            max_class_loss_batch = per_class_loss_batch.max()
            avg_class_loss_batch = per_class_loss_batch.mean()
            
            # Use previous epoch's average for penalty
            margin = 0.05
            avg_penalty = F.relu(avg_class_loss_batch - prev_avg_loss + margin)
            
            loss_power = max_class_loss_batch + 0 * avg_penalty
            #-----------------------------------------------------------------------------------------------------------
            loss_power = lambda_rate * loss_power            
            loss = loss_model + loss_power
            loss_model_sum += loss_model.item() * x.size(0)
            loss_power_sum += loss_power.item() * x.size(0)
            total += x.size(0)
            loss /= 100

        #-----------------------------------------------------------------------------------------------------------
        
        loss.backward()
        optimizer.step() 

        _, predicted_adv = x_hat.max(1)
        for class_id in range(num_classes):
            class_mask = (y == class_id)
            rob_correct[class_id] += (predicted_adv.eq(y) & class_mask).sum().item()

    # Calculate epoch-level metrics and update globals for next epoch
    if epoch > warmup_epochs:
        with torch.no_grad():
            per_class_loss_epoch = epoch_class_loss_sum / (epoch_class_count + 1e-6)
            max_class_loss_epoch = per_class_loss_epoch.max()
            avg_class_loss_epoch = per_class_loss_epoch.mean()
            
            # Update for next epoch
            prev_max_loss = max_class_loss_epoch.clone()
            prev_avg_loss = avg_class_loss_epoch.clone()

    rob_correct_perc = [rob / (50000 / num_classes) * 100 for rob in rob_correct]
    if epoch <= warmup_epochs:
        return rob_correct_perc, 0, 0
    else:
        return rob_correct_perc, loss_model_sum/total, loss_power_sum/total

def test():
    # For accuracies
    clean_correct = 0
    clean_correct_per_class = [0] * num_classes
    rob_correct = [0]*num_classes
    # For logits
    all_clean_logits = []
    all_rob_logits = []
    all_labels = []
    # For confusion matrix
    clean_confusion_matrix = torch.zeros(num_classes, num_classes, device=device)
    rob_confusion_matrix = torch.zeros(num_classes, num_classes, device=device)
    net.eval()
    with torch.no_grad():
        for batch_id, (x,y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            
            # Clean accuracy
            clean_logits = net(x)
            _, clean_predicted = clean_logits.max(1)
            clean_correct += clean_predicted.eq(y).sum().item()

            for class_id in range(num_classes):
                class_mask = (y == class_id)
                clean_correct_per_class[class_id] += (clean_predicted.eq(y) & class_mask).sum().item()

            # Clean Conf Matrix
            for i in range(len(y)):
                true_label, pred_label = y[i].item(), clean_predicted[i].item()
                clean_confusion_matrix[true_label, pred_label] += 1

            # Robust Accuracy
            x_adv = adversary.perturb_test(x, y)
            x_logits = net(x_adv)
            _, predicted = x_logits.max(1)

            for class_id in range(num_classes):
                class_mask = (y == class_id)
                rob_correct[class_id] += (predicted.eq(y) & class_mask).sum().item()

            # Robust Conf Matrix
            for i in range(len(y)):
                true_label, pred_label = y[i].item(), predicted[i].item()
                rob_confusion_matrix[true_label, pred_label] += 1

            # Saving Logits
            all_clean_logits.append(clean_logits.cpu())
            all_rob_logits.append(x_logits.cpu())
            all_labels.append(y.cpu())

        # Calculating accuracies
        clean_correct_perc = clean_correct / 10000 * 100
        clean_correct_perc_per_class = [clean / (10000 / num_classes) * 100 for clean in clean_correct_per_class]
        rob_correct_perc = [rob / (10000 / num_classes) * 100 for rob in rob_correct]

        # Calculating logits
        all_clean_logits = torch.cat(all_clean_logits, dim=0)  # (10000, 10)
        all_rob_logits = torch.cat(all_rob_logits, dim=0)      # (10000, 10)
        all_labels = torch.cat(all_labels, dim=0)              # (10000,)

        # Calculating confusion matrixes
        clean_row_sums = clean_confusion_matrix.sum(dim=1, keepdim=True).clamp(min=1e-8)
        clean_confusion_matrix = clean_confusion_matrix / clean_row_sums
        
        rob_row_sums = rob_confusion_matrix.sum(dim=1, keepdim=True).clamp(min=1e-8)
        rob_confusion_matrix = rob_confusion_matrix / rob_row_sums
        
    return (clean_correct_perc_per_class, rob_correct_perc, all_clean_logits, 
        all_rob_logits, all_labels, clean_confusion_matrix, rob_confusion_matrix)

def adjust_learning_rate(optimizer, epoch, initial):
    lr = initial
    if epoch >= destination_epoch  - 10:
        lr = lr * 0.1
    if epoch >= destination_epoch - 5:
        lr = lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr  

def evaluate_autoattack(net, test_loader, epsilon, checkpoint_path, results_path):
    '''
    Evaluate model using AutoAttack
    '''
    print(f"\n{'='*60}")
    print(f"Running AutoAttack evaluation on {checkpoint_path}")
    print(f"{'='*60}\n")
    
    # Load the best model
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    
    # Prepare data for AutoAttack (it needs all data at once or batch by batch)
    x_test = []
    y_test = []
    
    for x, y in test_loader:
        x_test.append(x)
        y_test.append(y)
    
    x_test = torch.cat(x_test, dim=0).to(device)
    y_test = torch.cat(y_test, dim=0).to(device)
    
    # Initialize AutoAttack
    adversary = AutoAttack(net, norm='Linf', eps=epsilon, version='standard', verbose=True)
    
    # Run AutoAttack
    print("Starting AutoAttack (this may take a while)...")
    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=128)
    
    # Evaluate per-class robust accuracy
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
    
    # Save results
    with open(f'{results_path}/AutoAttack_Results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Class', 'Robust_Accuracy'])
        for i, acc in enumerate(rob_correct_perc):
            writer.writerow([i, f'{acc:.2f}'])
        writer.writerow(['Average', f'{sum(rob_correct_perc)/len(rob_correct_perc):.2f}'])
        writer.writerow(['Min', f'{min(rob_correct_perc):.2f}'])
    
    print(f"\n{'='*60}")
    print("AutoAttack Results:")
    print(f"{'='*60}")
    for i, acc in enumerate(rob_correct_perc):
        print(f"Class {i}: {acc:.2f}%")
    print(f"Average: {sum(rob_correct_perc)/len(rob_correct_perc):.2f}%")
    print(f"Min: {min(rob_correct_perc):.2f}%")
    print(f"{'='*60}\n")
    
    return rob_correct_perc

def make_dir(model_type, lr, lam, target):
    '''
    Ensures that the directories exist for CSVs
    '''
    parent = file_specific_name
    script_name = Path(__file__).stem
    folder = f"{script_name}_{target}_{model_type}__LR{lr:.0e}__L{lam}"
    full_path = os.path.join("CSV",parent,script_name)
    os.makedirs(full_path, exist_ok=True)
    return full_path

def make_checkpoint_dir(model_type, lr, lam, target):
    '''
    Ensures that the checkpoint directory exists
    '''
    parent = file_specific_name
    script_name = Path(__file__).stem
    folder = f"{script_name}_{target}_{model_type}__LR{lr:.0e}__L{lam}"
    full_path = os.path.join("Checkpoints",parent,script_name)
    os.makedirs(full_path, exist_ok=True)
    return full_path

def save_checkpoint(state, checkpoint_dir, filename='checkpoint.pth'):
    '''
    Save checkpoint
    '''
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    print(f'Checkpoint saved to {filepath}')

for model_type, lr, lam, target_mode in product(model_types, lrs, lambdas, target_modes):
    print(f"---- New AN: model: {model_type}, lr: {lr}, lambda: {lam}, attack: {target_mode}")
    path = make_dir(model_type, lr, lam, target_mode)
    checkpoint_dir = make_checkpoint_dir(model_type, lr, lam, target_mode)

    # Reset the model
    net = ResNet18().to(device)
    net = nn.DataParallel(net)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0002)
    adversary = LinfPGDAttack(net)  
    
    # Reset global variables for each new run
    prev_max_loss = torch.tensor(float('inf'), device=device)
    prev_avg_loss = torch.tensor(float('inf'), device=device)
    best_min_acc = 0.0
    print(path)
    for epoch in range(destination_epoch):
        t0 = time.time()
        adjust_learning_rate(optimizer, epoch, lr)
        # Train CSV
        train_rob, train_loss_model, train_loss_power = train(epoch, lam, model_type)
        # Robust accuracies in Train
        with open(f'./{path}/Train_Rob.csv', 'a') as f:
            f.write(','.join(map(str, [format (x, '.1f') for x in train_rob])) + '\n')
        # Test CSV
        test_clean, test_rob, test_log_clean, test_log_rob, test_y, clean_mat, rob_mat  = test()
        # Test Robust
        with open(f'./{path}/Test_Rob.csv', 'a') as f:
            f.write(','.join(map(str, [format (x, '.1f') for x in test_rob])) + '\n')
        # Test Clean
        with open(f'./{path}/Test_Clean.csv', 'a') as f:
            f.write(','.join(map(str, [format (x, '.1f') for x in test_clean])) + '\n')


        # Logits Clean as of Last Test
        clean_logits_list = test_log_clean.tolist()
        with open(f'./{path}/Logits_Clean.csv', 'w') as f:
            for row in clean_logits_list:
                f.write(','.join([format(x, '.1f') for x in row]) + '\n')
        # Logits Robust as of Last Test
        rob_logits_list = test_log_rob.tolist()
        with open(f'./{path}/Logits_Rob.csv', 'w') as f:
             for row in rob_logits_list:
                f.write(','.join([format(x, '.1f') for x in row]) + '\n')
        # True Labels as of Last Test
        y_list = test_y.tolist()
        with open(f'./{path}/Logits_Labels.csv', 'w') as f:
            for label in y_list:  # Each label is just a number
                f.write(str(label) + '\n')



        # Loss CSV
        write_header = not os.path.exists(f'./{path}/Train_Loss.csv') or os.path.getsize(f'./{path}/Train_Loss.csv') == 0
        with open(f'./{path}/Train_Loss.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['train_loss_model', 'train_loss_power'])
            writer.writerow([train_loss_model, train_loss_power])

        current_min_acc = min(test_rob)
        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'Q': Q,
            'prev_max_loss': prev_max_loss,
            'prev_avg_loss': prev_avg_loss,
            'test_rob': test_rob,
            'best_min_acc': best_min_acc,
            'model_type': model_type,
            'lr': lr,
            'lambda': lam,
            'target_mode': target_mode
        }

        # if current_min_acc > best_min_acc:
        #     best_min_acc = current_min_acc
        #     save_checkpoint(checkpoint_state, checkpoint_dir, 'best_model.pth') 

        if epoch > (warmup_epochs + 10):
            save_checkpoint(checkpoint_state, checkpoint_dir, f"epoch_{epoch}_model.pth")

        #Clean conf matrix
        confusion_np = clean_mat.cpu().numpy()
        with open(f'{path}/confusion_matrix_clean_{epoch}.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['True/Pred'] + [f'Class_{i}' for i in range(num_classes)]
            writer.writerow(header)
            for i in range(num_classes):
                row = [f'Class_{i}'] + [f'{confusion_np[i, j]:.4f}' for j in range(num_classes)]
                writer.writerow(row)

        confusion_np = rob_mat.cpu().numpy()
        with open(f'{path}/confusion_matrix_robust_{epoch}.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['True/Pred'] + [f'Class_{i}' for i in range(num_classes)]
            writer.writerow(header)
            for i in range(num_classes):
                row = [f'Class_{i}'] + [f'{confusion_np[i, j]:.4f}' for j in range(num_classes)]
                writer.writerow(row)

        t1 = time.time()
        print(f'took {t1-t0:.1f} seconds.')
        
    # save_checkpoint(checkpoint_state, checkpoint_dir, f'best_model.pth')
    confusion_np = adversary.confusion_matrix.cpu().numpy()
    with open(f'{path}/confusion_matrix_train.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['True/Pred'] + [f'Class_{i}' for i in range(num_classes)]
        writer.writerow(header)
        for i in range(num_classes):
            row = [f'Class_{i}'] + [f'{confusion_np[i, j]:.4f}' for j in range(num_classes)]
            writer.writerow(row)
    
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        autoattack_results = evaluate_autoattack(
            net=net, 
            test_loader=test_loader,
            epsilon=epsilon,
            checkpoint_path=best_model_path,
            results_path=path
        )
    else:
        print(f"Warning: Best model not found at {best_model_path}")
    