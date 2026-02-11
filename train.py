"""
IBC-EDL Training Script
Combines IBC's soft label generation with EDL's evidential learning

Training Strategy:
1. Warmup (30 epochs): CE + balanced softmax
2. IBC (150 epochs): IBC soft labels + balanced softmax
3. EDL (20 epochs): EDL fine-tuning with soft labels
"""
import sys
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import argparse
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from tqdm import tqdm

from models import IBC_EDL_Model
from losses import edl_digamma_loss_soft, edl_log_loss_soft, edl_mse_loss_soft, relu_evidence
from dataloader import get_dataloader


def parse_args():
    parser = argparse.ArgumentParser(description='IBC-EDL Training')
    
    # Training
    parser.add_argument('--warm_up', default=30, type=int)
    parser.add_argument('--ibc_epochs', default=140, type=int)
    parser.add_argument('--edl_epochs', default=30, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.02, type=float)
    parser.add_argument('--gpuid', default=0, type=int)
    parser.add_argument('--seed', default=123, type=int)
    
    # Dataset
    parser.add_argument('--data_path', default='/mnt/zzh', type=str)
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100'])
    parser.add_argument('--imb_type', default='exp', type=str, choices=['exp', 'step'])
    parser.add_argument('--imb_factor', default=0.01, type=float)
    
    # IBC parameters
    parser.add_argument('--k', default=0.5, type=float, help='ratio of k nearest neighbors')
    parser.add_argument('--epsilon', default=0.2, type=float, help='soft label smoothing factor')
    parser.add_argument('--beta', default=0.99, type=float, help='EMA smoothing for prototypes')
    
    # EDL parameters
    parser.add_argument('--loss', default='edl_digamma', type=str,
                       choices=['edl_digamma', 'edl_log', 'edl_mse'])
    parser.add_argument('--skip_edl_train', action='store_true',
                       help='Skip EDL training stage, only use EDL for inference uncertainty')
    
    # Other
    parser.add_argument('--pretrained', default=None, type=str)
    parser.add_argument('--name', default=None, type=str)
    
    args = parser.parse_args()
    
    args.num_class = 100 if args.dataset == 'cifar100' else 10
    args.num_epochs = args.warm_up + args.ibc_epochs + args.edl_epochs
    if args.pretrained is None:
        args.pretrained = f"/mnt/zzh/moco_ckpt/PreActResNet18/{args.dataset}_exp_{args.imb_factor}/checkpoint_2000.pth.tar"
    if args.name is None:
        args.name = f'ibc_edl_{args.loss}_imb{args.imb_factor}'
    
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def balanced_softmax_loss(labels, logits, sample_per_class, reduction, tau=1.0):
    """Balanced softmax loss for class imbalance"""
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + tau * spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss


def warmup_train(epoch, model, optimizer, train_loader, args, cls_num_list):
    """Stage 1: Warmup with CE + balanced softmax"""
    model.train()
    total_loss = 0
    spc = torch.tensor(cls_num_list).cuda()
    
    pbar = tqdm(train_loader, desc=f'Warmup Epoch {epoch}')
    for inputs, labels, _ in pbar:
        inputs, labels = inputs.cuda(), labels.cuda()
        
        features, logits_main, logits_tail, logits_medium = model(inputs, return_features=True)
        
        loss_main = F.cross_entropy(logits_main, labels)
        loss_tail = balanced_softmax_loss(labels, logits_tail, spc, "mean", tau=0.5)
        loss_medium = balanced_softmax_loss(labels, logits_medium, spc, "mean", tau=1.0)
        
        loss = loss_main + loss_tail + loss_medium
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader)


def train_ibc(epoch, model, optimizer, train_loader, 
              head_nn_idx, medium_nn_idx, tail_nn_idx, args, cls_num_list):
    """Stage 2: IBC training with soft labels"""
    model.train()
    total_loss = 0
    spc = torch.tensor(cls_num_list).cuda()
    
    pbar = tqdm(train_loader, desc=f'IBC Epoch {epoch}')
    for inputs, labels, indices in pbar:
        inputs, labels = inputs.cuda(), labels.cuda()
        batch_size = inputs.size(0)
        
        # Generate soft labels
        targets_tail = torch.zeros(batch_size, args.num_class, device=inputs.device)
        targets_medium = torch.zeros(batch_size, args.num_class, device=inputs.device)
        
        targets_tail.scatter_(1, labels.unsqueeze(1), 1 - args.epsilon)
        targets_medium.scatter_(1, labels.unsqueeze(1), 1 - args.epsilon)
        
        for i in range(batch_size):
            idx = indices[i].item()
            
            tail_neighbors = tail_nn_idx[idx]
            if len(tail_neighbors) > 0:
                for n in tail_neighbors:
                    targets_tail[i, n] += args.epsilon / len(tail_neighbors)
            else:
                targets_tail[i, labels[i]] += args.epsilon
            
            medium_neighbors = medium_nn_idx[idx]
            if len(medium_neighbors) > 0:
                for n in medium_neighbors:
                    targets_medium[i, n] += args.epsilon / len(medium_neighbors)
            else:
                targets_medium[i, labels[i]] += args.epsilon
        
        targets_tail = targets_tail / targets_tail.sum(dim=1, keepdim=True)
        targets_medium = targets_medium / targets_medium.sum(dim=1, keepdim=True)
        
        # Forward
        features, logits_main, logits_tail, logits_medium = model(inputs, return_features=True)
        
        # Losses
        loss_main = F.cross_entropy(logits_main, labels)
        
        adjusted_logits_tail = logits_tail + 1.0 * spc.log()
        loss_tail = -torch.mean(torch.sum(F.log_softmax(adjusted_logits_tail, dim=1) * targets_tail, dim=1))
        
        adjusted_logits_medium = logits_medium + 0.5 * spc.log()
        loss_medium = -torch.mean(torch.sum(F.log_softmax(adjusted_logits_medium, dim=1) * targets_medium, dim=1))
        
        # Regularization
        prior = torch.ones(args.num_class).cuda() / args.num_class
        pred_mean = torch.softmax(logits_main, dim=1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean))
        
        loss = loss_main + loss_tail + loss_medium + penalty
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader)


def train_edl(epoch, model, optimizer, train_loader,
              head_nn_idx, medium_nn_idx, tail_nn_idx, args, cls_num_list):
    """Stage 3: EDL fine-tuning"""
    model.train()
    total_loss = 0
    spc = torch.tensor(cls_num_list).cuda()
    
    loss_funcs = {
        'edl_digamma': edl_digamma_loss_soft,
        'edl_log': edl_log_loss_soft,
        'edl_mse': edl_mse_loss_soft,
    }
    criterion = loss_funcs[args.loss]
    
    pbar = tqdm(train_loader, desc=f'EDL Epoch {epoch}')
    for inputs, labels, indices in pbar:
        inputs, labels = inputs.cuda(), labels.cuda()
        batch_size = inputs.size(0)
        
        # Generate soft labels
        targets_main = torch.zeros(batch_size, args.num_class, device=inputs.device)
        targets_tail = torch.zeros(batch_size, args.num_class, device=inputs.device)
        targets_medium = torch.zeros(batch_size, args.num_class, device=inputs.device)
        
        targets_main.scatter_(1, labels.unsqueeze(1), 1 - args.epsilon)
        targets_tail.scatter_(1, labels.unsqueeze(1), 1 - args.epsilon)
        targets_medium.scatter_(1, labels.unsqueeze(1), 1 - args.epsilon)
        
        for i in range(batch_size):
            idx = indices[i].item()
            
            head_neighbors = head_nn_idx[idx]
            if len(head_neighbors) > 0:
                for n in head_neighbors:
                    targets_main[i, n] += args.epsilon / len(head_neighbors)
            else:
                targets_main[i, labels[i]] += args.epsilon
            
            tail_neighbors = tail_nn_idx[idx]
            if len(tail_neighbors) > 0:
                for n in tail_neighbors:
                    targets_tail[i, n] += args.epsilon / len(tail_neighbors)
            else:
                targets_tail[i, labels[i]] += args.epsilon
            
            medium_neighbors = medium_nn_idx[idx]
            if len(medium_neighbors) > 0:
                for n in medium_neighbors:
                    targets_medium[i, n] += args.epsilon / len(medium_neighbors)
            else:
                targets_medium[i, labels[i]] += args.epsilon
        
        targets_main = targets_main / targets_main.sum(dim=1, keepdim=True)
        targets_tail = targets_tail / targets_tail.sum(dim=1, keepdim=True)
        targets_medium = targets_medium / targets_medium.sum(dim=1, keepdim=True)
        
        # Forward
        features, logits_main, logits_tail, logits_medium = model(inputs, return_features=True)
        
        # Apply balanced adjustment
        adjusted_logits_tail = logits_tail + 1.0 * spc.log()
        adjusted_logits_medium = logits_medium + 0.5 * spc.log()
        
        # EDL losses
        loss_main = criterion(logits_main, targets_main, epoch, args.num_class, 10, inputs.device)
        loss_tail = criterion(adjusted_logits_tail, targets_tail, epoch, args.num_class, 10, inputs.device)
        loss_medium = criterion(adjusted_logits_medium, targets_medium, epoch, args.num_class, 10, inputs.device)
        
        loss = loss_main + loss_tail + loss_medium
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader)



def extract_features_and_compute_neighbors(model, train_loader, args, 
                                           many_shot_classes, medium_shot_classes, few_shot_classes,
                                           cfeats_EMA, epoch):
    """Extract features and compute KNN neighbors"""
    model.eval()
    
    all_features = []
    all_labels = []
    all_indices = []
    
    with torch.no_grad():
        for inputs, labels, indices in tqdm(train_loader, desc='Extracting features'):
            inputs = inputs.cuda()
            features, _, _, _ = model(inputs, return_features=True)
            all_features.append(features.cpu())
            all_labels.append(labels)
            all_indices.append(indices)
    
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_indices = torch.cat(all_indices, dim=0)
    
    sorted_idx = torch.argsort(all_indices)
    all_features = all_features[sorted_idx]
    all_labels = all_labels[sorted_idx]
    
    # Compute centroids
    centroids = []
    for c in range(args.num_class):
        mask = (all_labels == c)
        if mask.sum() > 0:
            centroids.append(all_features[mask].mean(dim=0))
        else:
            centroids.append(torch.zeros(512))
    centroids = torch.stack(centroids).numpy().astype(np.float32)
    
    # Update EMA
    if epoch == args.warm_up + 1:
        cfeats_EMA = centroids.copy()
    else:
        cfeats_EMA = args.beta * cfeats_EMA + (1 - args.beta) * centroids
    
    # Normalize
    features_np = all_features.numpy().astype(np.float32)
    features_np = features_np - features_np.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(features_np, axis=1, keepdims=True) + 1e-8
    features_np = features_np / norms
    
    centroids_norm = cfeats_EMA - cfeats_EMA.mean(axis=1, keepdims=True)
    centroids_norms = np.linalg.norm(centroids_norm, axis=1, keepdims=True) + 1e-8
    centroids_norm = centroids_norm / centroids_norms
    
    # FAISS search
    k = int(args.num_class * args.k)
    index = faiss.IndexFlatIP(centroids_norm.shape[1])
    index.add(centroids_norm.astype(np.float32))
    _, nn_indices = index.search(features_np, k)
    
    # Categorize neighbors
    head_nn_idx = []
    medium_nn_idx = []
    tail_nn_idx = []
    
    for i in range(len(nn_indices)):
        neighbors = nn_indices[i]
        head_nn_idx.append([n for n in neighbors if n in many_shot_classes])
        medium_nn_idx.append([n for n in neighbors if n in medium_shot_classes])
        tail_nn_idx.append([n for n in neighbors if n in few_shot_classes])
    
    return cfeats_EMA, head_nn_idx, medium_nn_idx, tail_nn_idx


def test(model, test_loader, args, many_shot_classes, medium_shot_classes, few_shot_classes, stage='edl'):
    """Test with ensemble of three experts"""
    model.eval()
    
    correct = 0
    total = 0
    many_correct, many_total = 0, 0
    medium_correct, medium_total = 0, 0
    few_correct, few_total = 0, 0
    
    class_uncertainty = torch.zeros(args.num_class)
    class_count = torch.zeros(args.num_class)
    
    use_edl = (stage == 'edl')
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Testing'):
            inputs, targets = inputs.cuda(), targets.cuda()
            features, logits_main, logits_tail, logits_medium = model(inputs, return_features=True)
            
            if use_edl:
                # EDL: use alpha
                alpha_main = relu_evidence(logits_main) + 1
                alpha_tail = relu_evidence(logits_tail) + 1
                alpha_medium = relu_evidence(logits_medium) + 1
                outputs = alpha_main + alpha_tail + alpha_medium
                S = outputs.sum(dim=1)
                uncertainty = (3 * args.num_class) / S
            else:
                # CE: use logits
                outputs = logits_main + logits_tail + logits_medium
                probs = F.softmax(outputs, dim=1)
                uncertainty = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            for i in range(len(targets)):
                t = targets[i].item()
                class_count[t] += 1
                class_uncertainty[t] += uncertainty[i].cpu().item()
                
                if t in many_shot_classes:
                    many_total += 1
                    if predicted[i] == targets[i]:
                        many_correct += 1
                elif t in medium_shot_classes:
                    medium_total += 1
                    if predicted[i] == targets[i]:
                        medium_correct += 1
                elif t in few_shot_classes:
                    few_total += 1
                    if predicted[i] == targets[i]:
                        few_correct += 1
    
    overall_acc = 100. * correct / total
    many_acc = 100. * many_correct / many_total if many_total > 0 else 0
    medium_acc = 100. * medium_correct / medium_total if medium_total > 0 else 0
    few_acc = 100. * few_correct / few_total if few_total > 0 else 0
    
    class_uncertainty = class_uncertainty / (class_count + 1e-8)
    many_unc = np.mean([class_uncertainty[c].item() for c in many_shot_classes])
    medium_unc = np.mean([class_uncertainty[c].item() for c in medium_shot_classes])
    few_unc = np.mean([class_uncertainty[c].item() for c in few_shot_classes])
    
    return {
        'overall_acc': overall_acc,
        'many_acc': many_acc,
        'medium_acc': medium_acc,
        'few_acc': few_acc,
        'many_unc': many_unc,
        'medium_unc': medium_unc,
        'few_unc': few_unc,
    }


def test_with_expert_uncertainty(model, test_loader, args, many_shot_classes, medium_shot_classes, few_shot_classes):
    """Detailed test with per-expert uncertainty analysis"""
    model.eval()
    
    total = 0
    ensemble_correct = 0
    expert_correct = {'main': 0, 'tail': 0, 'medium': 0}
    
    expert_uncertainty_sum = {
        'main': torch.zeros(args.num_class),
        'tail': torch.zeros(args.num_class),
        'medium': torch.zeros(args.num_class),
        'ensemble': torch.zeros(args.num_class)
    }
    class_count = torch.zeros(args.num_class)
    
    shot_stats = {
        'many': {'total': 0, 'correct': {'main': 0, 'tail': 0, 'medium': 0, 'ensemble': 0}},
        'medium': {'total': 0, 'correct': {'main': 0, 'tail': 0, 'medium': 0, 'ensemble': 0}},
        'few': {'total': 0, 'correct': {'main': 0, 'tail': 0, 'medium': 0, 'ensemble': 0}}
    }
    
    all_agree = 0
    two_agree = 0
    all_disagree = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Expert Analysis'):
            inputs, targets = inputs.cuda(), targets.cuda()
            batch_size = inputs.size(0)
            
            features, logits_main, logits_tail, logits_medium = model(inputs, return_features=True)
            
            alpha_main = relu_evidence(logits_main) + 1
            alpha_tail = relu_evidence(logits_tail) + 1
            alpha_medium = relu_evidence(logits_medium) + 1
            
            _, pred_main = alpha_main.max(1)
            _, pred_tail = alpha_tail.max(1)
            _, pred_medium = alpha_medium.max(1)
            
            unc_main = args.num_class / alpha_main.sum(dim=1)
            unc_tail = args.num_class / alpha_tail.sum(dim=1)
            unc_medium = args.num_class / alpha_medium.sum(dim=1)
            
            alpha_ensemble = alpha_main + alpha_tail + alpha_medium
            _, pred_ensemble = alpha_ensemble.max(1)
            unc_ensemble = args.num_class / alpha_ensemble.sum(dim=1)
            
            total += batch_size
            ensemble_correct += pred_ensemble.eq(targets).sum().item()
            expert_correct['main'] += pred_main.eq(targets).sum().item()
            expert_correct['tail'] += pred_tail.eq(targets).sum().item()
            expert_correct['medium'] += pred_medium.eq(targets).sum().item()
            
            for i in range(batch_size):
                t = targets[i].item()
                class_count[t] += 1
                
                expert_uncertainty_sum['main'][t] += unc_main[i].cpu().item()
                expert_uncertainty_sum['tail'][t] += unc_tail[i].cpu().item()
                expert_uncertainty_sum['medium'][t] += unc_medium[i].cpu().item()
                expert_uncertainty_sum['ensemble'][t] += unc_ensemble[i].cpu().item()
                
                if t in many_shot_classes:
                    shot_type = 'many'
                elif t in medium_shot_classes:
                    shot_type = 'medium'
                else:
                    shot_type = 'few'
                
                shot_stats[shot_type]['total'] += 1
                if pred_main[i] == targets[i]:
                    shot_stats[shot_type]['correct']['main'] += 1
                if pred_tail[i] == targets[i]:
                    shot_stats[shot_type]['correct']['tail'] += 1
                if pred_medium[i] == targets[i]:
                    shot_stats[shot_type]['correct']['medium'] += 1
                if pred_ensemble[i] == targets[i]:
                    shot_stats[shot_type]['correct']['ensemble'] += 1
                
                preds = [pred_main[i].item(), pred_tail[i].item(), pred_medium[i].item()]
                if preds[0] == preds[1] == preds[2]:
                    all_agree += 1
                elif preds[0] == preds[1] or preds[1] == preds[2] or preds[0] == preds[2]:
                    two_agree += 1
                else:
                    all_disagree += 1
    
    for key in expert_uncertainty_sum:
        expert_uncertainty_sum[key] = expert_uncertainty_sum[key] / (class_count + 1e-8)
    
    def compute_shot_uncertainty(unc_tensor, shot_classes):
        return np.mean([unc_tensor[c].item() for c in shot_classes]) if len(shot_classes) > 0 else 0
    
    results = {
        'overall_acc': 100. * ensemble_correct / total,
        'expert_acc': {
            'main': 100. * expert_correct['main'] / total,
            'tail': 100. * expert_correct['tail'] / total,
            'medium': 100. * expert_correct['medium'] / total,
        },
        'shot_acc': {},
        'shot_uncertainty': {},
        'agreement': {
            'all_agree': 100. * all_agree / total,
            'two_agree': 100. * two_agree / total,
            'all_disagree': 100. * all_disagree / total,
        }
    }
    
    for shot_type in ['many', 'medium', 'few']:
        shot_total = shot_stats[shot_type]['total']
        if shot_total > 0:
            results['shot_acc'][shot_type] = {
                'main': 100. * shot_stats[shot_type]['correct']['main'] / shot_total,
                'tail': 100. * shot_stats[shot_type]['correct']['tail'] / shot_total,
                'medium': 100. * shot_stats[shot_type]['correct']['medium'] / shot_total,
                'ensemble': 100. * shot_stats[shot_type]['correct']['ensemble'] / shot_total,
            }
        else:
            results['shot_acc'][shot_type] = {'main': 0, 'tail': 0, 'medium': 0, 'ensemble': 0}
        
        shot_classes = many_shot_classes if shot_type == 'many' else (
            medium_shot_classes if shot_type == 'medium' else few_shot_classes
        )
        results['shot_uncertainty'][shot_type] = {
            'main': compute_shot_uncertainty(expert_uncertainty_sum['main'], shot_classes),
            'tail': compute_shot_uncertainty(expert_uncertainty_sum['tail'], shot_classes),
            'medium': compute_shot_uncertainty(expert_uncertainty_sum['medium'], shot_classes),
            'ensemble': compute_shot_uncertainty(expert_uncertainty_sum['ensemble'], shot_classes),
        }
    
    return results


def print_expert_analysis(results):
    """Pretty print expert analysis"""
    print("\n" + "=" * 70)
    print("EXPERT ANALYSIS")
    print("=" * 70)
    
    print(f"\n[Overall Accuracy]")
    print(f"  Ensemble: {results['overall_acc']:.2f}%")
    print(f"  Main:     {results['expert_acc']['main']:.2f}%")
    print(f"  Tail:     {results['expert_acc']['tail']:.2f}%")
    print(f"  Medium:   {results['expert_acc']['medium']:.2f}%")
    
    print(f"\n[Accuracy by Shot Type]")
    print(f"{'':12} {'Ensemble':>10} {'Main':>10} {'Tail':>10} {'Medium':>10}")
    print("-" * 54)
    for shot_type in ['many', 'medium', 'few']:
        acc = results['shot_acc'][shot_type]
        print(f"{shot_type.capitalize():12} {acc['ensemble']:>10.2f} {acc['main']:>10.2f} "
              f"{acc['tail']:>10.2f} {acc['medium']:>10.2f}")
    
    print(f"\n[Uncertainty by Shot Type]")
    print(f"{'':12} {'Ensemble':>10} {'Main':>10} {'Tail':>10} {'Medium':>10}")
    print("-" * 54)
    for shot_type in ['many', 'medium', 'few']:
        unc = results['shot_uncertainty'][shot_type]
        print(f"{shot_type.capitalize():12} {unc['ensemble']:>10.4f} {unc['main']:>10.4f} "
              f"{unc['tail']:>10.4f} {unc['medium']:>10.4f}")
    
    print(f"\n[Expert Agreement]")
    print(f"  All 3 agree:    {results['agreement']['all_agree']:.2f}%")
    print(f"  2 agree:        {results['agreement']['two_agree']:.2f}%")
    print(f"  All disagree:   {results['agreement']['all_disagree']:.2f}%")
    print("=" * 70 + "\n")



def main():
    args = parse_args()
    print(args)
    
    torch.cuda.set_device(args.gpuid)
    set_seed(args.seed)
    
    # Load data
    print('\n=== Loading Data ===')
    train_loader, test_loader, cls_num_list = get_dataloader(
        dataset=args.dataset,
        imb_type=args.imb_type,
        imb_factor=args.imb_factor,
        batch_size=args.batch_size,
        num_workers=4,
        root_dir=args.data_path
    )
    
    # Identify shot types
    sorted_classes = sorted(enumerate(cls_num_list), key=lambda x: x[1], reverse=True)
    sorted_classes = [c for c, _ in sorted_classes]
    n_classes = len(cls_num_list)
    top_30 = int(n_classes * 0.3)
    bottom_30 = int(n_classes * 0.3)
    
    many_shot_classes = set(sorted_classes[:top_30])
    few_shot_classes = set(sorted_classes[-bottom_30:])
    medium_shot_classes = set(sorted_classes[top_30:-bottom_30]) if top_30 < n_classes - bottom_30 else set()
    
    print(f'Many: {len(many_shot_classes)}, Medium: {len(medium_shot_classes)}, Few: {len(few_shot_classes)}')
    
    # Create model
    print('\n=== Building Model ===')
    model = IBC_EDL_Model(num_classes=args.num_class)
    model = model.cuda()
    cudnn.benchmark = True
    
    # Load MoCo weights
    if os.path.isfile(args.pretrained):
        print(f"Loading MoCo: {args.pretrained}")
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        state_dict = checkpoint['state_dict']
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('encoder_q') and not k.startswith('encoder_q.fc'):
                new_key = k.replace('encoder_q.', '')
                new_state_dict[new_key] = v
        
        msg = model.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded (missing: {len(msg.missing_keys)}, unexpected: {len(msg.unexpected_keys)})")
    else:
        print(f"No MoCo checkpoint found, training from scratch")
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
    cfeats_EMA = np.zeros((args.num_class, 512), dtype=np.float32)
    
    warmup_end = args.warm_up
    ibc_end = args.warm_up + args.ibc_epochs
    total_epochs = args.num_epochs
    
    print(f'\n=== Training Schedule ===')
    print(f'Warmup: 1-{warmup_end}, IBC: {warmup_end+1}-{ibc_end}, EDL: {ibc_end+1}-{total_epochs}')
    
    print('\n=== Start Training ===')
    best_acc = 0
    best_epoch = 0
    results = []
    
    for epoch in range(1, total_epochs + 1):
        # Learning rate
        lr = args.lr
        if epoch > 150:
            lr /= 10
        if epoch > ibc_end:
            lr = args.lr / 100
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Stage
        if epoch <= warmup_end:
            stage = 'warmup'
        elif epoch <= ibc_end:
            stage = 'ibc'
        else:
            stage = 'edl'
        
        # Train
        if stage == 'warmup':
            train_loss = warmup_train(epoch, model, optimizer, train_loader, args, cls_num_list)
            if epoch == warmup_end:
                cfeats_EMA, head_nn_idx, medium_nn_idx, tail_nn_idx = \
                    extract_features_and_compute_neighbors(
                        model, train_loader, args,
                        many_shot_classes, medium_shot_classes, few_shot_classes,
                        cfeats_EMA, epoch
                    )
        elif stage == 'ibc':
            cfeats_EMA, head_nn_idx, medium_nn_idx, tail_nn_idx = \
                extract_features_and_compute_neighbors(
                    model, train_loader, args,
                    many_shot_classes, medium_shot_classes, few_shot_classes,
                    cfeats_EMA, epoch
                )
            train_loss = train_ibc(
                epoch, model, optimizer, train_loader,
                head_nn_idx, medium_nn_idx, tail_nn_idx, args, cls_num_list
            )
        else:
            cfeats_EMA, head_nn_idx, medium_nn_idx, tail_nn_idx = \
                extract_features_and_compute_neighbors(
                    model, train_loader, args,
                    many_shot_classes, medium_shot_classes, few_shot_classes,
                    cfeats_EMA, epoch
                )
            train_loss = train_edl(
                epoch, model, optimizer, train_loader,
                head_nn_idx, medium_nn_idx, tail_nn_idx, args, cls_num_list
            )
        
        # Test
        test_results = test(model, test_loader, args, 
                           many_shot_classes, medium_shot_classes, few_shot_classes, stage)
        
        print(f"\nEpoch {epoch}/{total_epochs} [{stage.upper()}]")
        print(f"Loss: {train_loss:.4f} | Acc: {test_results['overall_acc']:.2f}% | "
              f"Many: {test_results['many_acc']:.2f}% | Med: {test_results['medium_acc']:.2f}% | "
              f"Few: {test_results['few_acc']:.2f}%")
        print(f"Unc - Many: {test_results['many_unc']:.4f} | Med: {test_results['medium_unc']:.4f} | "
              f"Few: {test_results['few_unc']:.4f} | Best: {best_acc:.2f}%@{best_epoch}")
        
        results.append({
            'epoch': epoch, 'stage': stage, 'train_loss': train_loss,
            'overall_acc': test_results['overall_acc'],
            'many_acc': test_results['many_acc'],
            'medium_acc': test_results['medium_acc'],
            'few_acc': test_results['few_acc'],
            'many_unc': test_results['many_unc'],
            'medium_unc': test_results['medium_unc'],
            'few_unc': test_results['few_unc'],
        })
        
        if test_results['overall_acc'] > best_acc:
            best_acc = test_results['overall_acc']
            best_epoch = epoch
            save_dir = Path(f'./checkpoint/{args.name}')
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch, 'acc': best_acc,
            }, save_dir / 'best_model.pth')
        
        if epoch in [warmup_end, ibc_end, total_epochs]:
            save_dir = Path(f'./checkpoint/{args.name}')
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
            }, save_dir / f'checkpoint_ep{epoch}_{stage}.pth')
    
    # Save results
    results_dir = Path(f'./results/{args.name}')
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(results_dir / 'results.csv', index=False)
    
    print(f'\n{"="*60}')
    print(f'Training completed! Best: {best_acc:.2f}% @ epoch {best_epoch}')
    print(f'{"="*60}')
    
    # Expert analysis
    print('\nRunning expert analysis...')
    expert_results = test_with_expert_uncertainty(
        model, test_loader, args,
        many_shot_classes, medium_shot_classes, few_shot_classes
    )
    print_expert_analysis(expert_results)
    
    import json
    with open(results_dir / 'expert_analysis.json', 'w') as f:
        json.dump(expert_results, f, indent=2)


if __name__ == '__main__':
    main()
