import math
import wandb
import torch
import torch.nn.functional as F
import numpy as np

from loguru import logger
from CAL.algorithm.sl import SLModel
from CAL.utils import AverageMeter, update_meter


class CAL_OWDFA(SLModel):
    def __init__(self, args):
        super(CAL_OWDFA, self).__init__(args)

        # 1) Basic configuration
        self.student_temp = args.student_temp
        self.teacher_temp = args.teacher_temp
        self.max_epochs = args.epochs # Total training epochs (used by scheduling in some losses)
        self.num_known = len(args.known_classes) # Number of known classes (used to split known/novel predictions)

        # 2) Loss hyper-parameters
        # CCR loss 
        self.ccr_weight = args.ccr_weight

        # ACR loss
        self.pseudo_epohcs = args.pseudo_epohcs   # start epoch for pseudo supervision
        self.known_conf = args.gamma              # confidence threshold for known predictions
        self.novel_conf = 0.0                     # confidence threshold for novel predictions (adaptive)


        # 3) Runtime statistics for adaptive thresholds
        # Track mean confidence of (selected) known vs novel predictions per epoch
        self._sum_conf_known = 0.0
        self._sum_conf_novel = 0.0
        self._cnt_known = 0
        self._cnt_novel = 0

        # Cache logits for per-epoch prototype usage analysis / merging
        self.logits_cache = []


        # 4) Optional: unknown category number estimation / prototype merging
        self.unlabeled_usage_coverage = args.unlabeled_usage_coverage
        self.enable_proto_pruning = args.enable_proto_pruning
        logger.info(f"Estimation enabled (est_k): {self.enable_proto_pruning}")
        logger.info(f"Configured model_num_classes: {args.model_num_classes}")

    
    # Return the list of loss names used during training.
    def get_loss_names(self):
        loss_name = [
            'total_loss',
            'cls_loss',
            'regularization_loss',
            'acr_loss',
            'ccr_loss',
        ]
        return loss_name
    

    # Hook called at the beginning of each training epoch.
    def on_train_epoch_start(self):
        self._sum_conf_known = 0.0
        self._sum_conf_novel = 0.0
        self._cnt_known      = 0
        self._cnt_novel      = 0
        self.train_losses = {loss: AverageMeter(loss, ':.2f') for loss in self.get_loss_names()}


    def training_step(self, batch, batch_idx):

        # Log current learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', current_lr, on_step=True, on_epoch=False, prog_bar=True)
        
        loss_map = {} # Container for loss values (filled later)

        # Unpack batch contents
        tag = batch['tag']
        tag = tag.squeeze()
        images = batch['image']
        targets = batch['target']

        # Forward pass: weak and strong views
        logits_weak = self.encoder(images[0])
        logits_strong = self.encoder(images[1])

        # Cache weak-branch logits for prototype analysis / merging
        self.logits_cache.append(logits_weak.detach())


        # Convert weak logits to probabilities and obtain predictions
        probs_weak = logits_weak.softmax(dim=-1)         # [B, C]
        max_probs_weak, pred_weak = probs_weak.max(dim=-1)  # [B], [B]

        # Agreement-based sample selection
        pred_strong = logits_strong.argmax(dim=-1)
        agree_mask = pred_weak.eq(pred_strong)
        sel_mask = agree_mask

        # Split selected samples into known vs. novel predictions
        known_idx = (pred_weak < self.num_known) & sel_mask
        novel_idx = (pred_weak >= self.num_known) & sel_mask

        # Accumulate confidence statistics
        self._sum_conf_known += max_probs_weak[known_idx].sum().item()
        self._cnt_known += known_idx.sum().item()
        self._sum_conf_novel += max_probs_weak[novel_idx].sum().item()
        self._cnt_novel += novel_idx.sum().item()


        # ============================================================
        # (1) Supervised classification loss
        #   - Applied only to labelled samples (tag == 1)
        #   - Both weak and strong views are supervised
        # ============================================================
        labeled_mask = (tag == 1)  # [B] boolean mask for labelled samples

        if torch.any(labeled_mask):
            # Extract logits of labelled samples from both views
            labeled_v1_logits = logits_weak[labeled_mask] / self.student_temp  # [L, num_classes]
            labeled_v2_logits = logits_strong[labeled_mask] / self.student_temp
            
            # Standard cross-entropy loss
            sup_logits = torch.cat([labeled_v1_logits, labeled_v2_logits], dim=0)
            sup_labels = torch.cat([targets[labeled_mask]] * 2, dim=0)
            cls_loss = F.cross_entropy(sup_logits, sup_labels)
        else:
            cls_loss = torch.tensor(0.0, device=self.device)

        # ============================================================
        # (2) Regularization term
        #   - Encourages balanced class usage across the batch
        #   - Prevents prototype collapse in open-world setting
        # ============================================================
        regularization_loss = self.regularization_loss([logits_weak, logits_strong], self.student_temp)  

        # ============================================================
        # (3) Confidence-aware consistency regularization
        # ============================================================
        ccr_loss = self.ccr_loss(logits_weak, logits_strong, self.current_epoch)

        # ============================================================
        # (4) Asymmetric Confidence Reinforcement (Pseudo-label supervision for unlabelled samples)
        #   - Activated only after a warm-up period (epoch >= pseudo_epohcs)
        #   - Uses class-specific confidence thresholds (known vs novel)
        # ============================================================
        acr_loss = torch.tensor(0.0, device=self.device)

        if self.current_epoch >= self.pseudo_epohcs:
            unlabeled_mask = (tag == 2)  # [B] boolean mask for unlabelled samples
            valid_mask = None
            if torch.any(unlabeled_mask):
                # Weak-view predictions are used to generate pseudo labels
                logits_u = logits_weak[unlabeled_mask]
                probs_u = F.softmax(logits_u / self.student_temp, dim=1)
                max_probs, pred_labels = probs_u.max(dim=1)

                # Separate predicted known vs novel samples
                known_pred_mask = pred_labels < self.num_known
                novel_pred_mask = ~known_pred_mask

                # Initialize all pseudo-labels as invalid (-1)
                pseudo_targets = torch.full_like(pred_labels, fill_value=-1)

                # Assign pseudo labels based on class-specific confidence thresholds
                pseudo_targets[(known_pred_mask) & (max_probs >= self.known_conf)] = pred_labels[(known_pred_mask) & (max_probs >= self.known_conf)]
                pseudo_targets[(novel_pred_mask) & (max_probs >= self.novel_conf)] = pred_labels[(novel_pred_mask) & (max_probs >= self.novel_conf)]

                # Select samples with valid pseudo labels
                valid_mask = pseudo_targets != -1

            # Compute pseudo-label classification loss only if valid samples exist
            if valid_mask is not None and valid_mask.any():
                num_pseudo = valid_mask.sum().item()
                self.log('pseudo/num_samples', num_pseudo, on_step=True, on_epoch=True, prog_bar=True)

                logits_valid = logits_u[valid_mask]
                targets_valid = pseudo_targets[valid_mask]
                acr_loss = F.cross_entropy(logits_valid, targets_valid)


        # Total loss
        total_loss = cls_loss + regularization_loss + acr_loss + self.ccr_weight * ccr_loss

        loss_map['total_loss'] = total_loss
        loss_map['cls_loss'] = cls_loss
        loss_map['regularization_loss'] = regularization_loss
        loss_map['acr_loss'] = acr_loss
        loss_map['ccr_loss'] = ccr_loss

        for key, value in loss_map.items():
            update_meter(
                self.train_losses[key], value, self.args.train.batch_size)
        for ls in self.train_losses.values():
            self.log(ls.name, ls.avg, on_step=True, prog_bar=True)
            
        return total_loss

    def on_train_epoch_end(self):
        self._log_epoch_losses()
        self._update_adaptive_thresholds()
        self._maybe_merge_prototypes_end_of_epoch()



    def ccr_loss(self, weak_output, strong_output, epoch, wmin=0.2, gamma_max=1): 

        probs      = F.softmax(weak_output / self.teacher_temp, dim=-1)
        max_probs, _ = probs.max(dim=-1)                # [B]

        lambda_t= (epoch+1) / self.max_epochs
        weight = (1 - lambda_t) * max_probs + lambda_t * (1.0 - max_probs)

        weight = weight.detach()

        teacher_logits_sel = weak_output / self.teacher_temp      
        teacher_probs      = F.softmax(teacher_logits_sel, dim=-1).detach()  # [M, C]

        student_logits_sel = strong_output / self.student_temp
        student_log_probs  = F.log_softmax(student_logits_sel, dim=-1)       # [M, C]

        per_sample_kl = (-teacher_probs * student_log_probs).sum(dim=-1)  # [M]

        loss = (per_sample_kl * weight).mean()

        return loss


    def regularization_loss(self, logits_list, student_temp):

        assert len(logits_list) >= 1
        V = len(logits_list)

        all_outputs = torch.cat(logits_list, dim=0)
        avg_probs = (all_outputs / student_temp).softmax(dim=1).mean(dim=0)
        regularization_loss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))
        return regularization_loss / V

    def _log_epoch_losses(self):
        """Log averaged training losses for the current epoch."""
        results = {key: meter.avg for key, meter in self.train_losses.items()}
        if self.args.use_wandb:
            wandb.log(results, step=self.current_epoch)

    def _update_adaptive_thresholds(self):
        """
        Adaptive confidence threshold calibration (Known vs Novel):
        - Compute mean confidence of selected known/novel predictions
        - Update novel_conf by scaling known_conf with their ratio
        """
        # ---- mean confidence for known predictions ----
        mean_conf_known = (self._sum_conf_known / self._cnt_known) if (self._cnt_known > 0) else None

        # ---- mean confidence for novel predictions ----
        mean_conf_novel = (self._sum_conf_novel / self._cnt_novel) if (self._cnt_novel > 0) else None

        # ---- update novel threshold ----
        if (mean_conf_known is not None) and (mean_conf_novel is not None):
            conf_ratio = mean_conf_novel / mean_conf_known
            conf_ratio = float(torch.clamp(torch.tensor(conf_ratio), min=0.1, max=10.0))

            self.novel_conf = self.known_conf * conf_ratio

            # log thresholds
            self.log("thr_known", self.known_conf, prog_bar=True)
            self.log("thr_novel", self.novel_conf, prog_bar=True)

            if self.args.use_wandb:
                wandb.log(
                    {"thr_known": self.known_conf, "thr_novel": self.novel_conf},
                    step=self.current_epoch,
                )

    def _maybe_merge_prototypes_end_of_epoch(self):
        """
        Prototype merging for unknown-K (optional):
        - Collect epoch logits (cached in training_step)
        - Analyze prototype usage
        - Merge low-usage unlabeled prototypes into a smaller set of anchors
        - Write merged prototypes back to encoder.fc2 and update valid_proto_num
        Always clears logits_cache at the end.
        """
        try:
            if not self.enable_proto_pruning:
                return

            all_logits = self._collect_epoch_logits()
            if all_logits is None:
                return

            # Analyze prototype usage statistics
            proto_weight = self.encoder.fc2[: self.encoder.valid_proto_num].detach()
            usage_count = self._analyze_prototypes_from_logits(
                all_logits.detach(),
                proto_weight,
                epoch=self.current_epoch,
            )

            logger.info(f"Before merging: fc2 shape = {self.encoder.fc2.shape}")

            # Merge unlabeled prototypes (keep known prototypes untouched)
            new_fc2 = self._merge_unlabeled_prototypes(
                fc_weight=proto_weight,
                usage_count=usage_count,
                num_known_protos=self.num_known,
                unlabeled_usage_coverage=self.unlabeled_usage_coverage,
            )

            # Write back only when merged prototype count is smaller
            new_num = new_fc2.shape[0]
            if new_num < self.encoder.valid_proto_num:
                with torch.no_grad():
                    self.encoder.fc2[:new_num].copy_(new_fc2)
                    self.encoder.fc2[new_num:self.encoder.valid_proto_num].zero_()
                    self.encoder.valid_proto_num = new_num

            logger.info(f"After merging: valid prototypes = {self.encoder.valid_proto_num}")

        finally:
            # Always clear cached logits at epoch end
            self.logits_cache.clear()


    
    def _collect_epoch_logits(self):
        if len(self.logits_cache) == 0:
            return None
        return torch.cat(self.logits_cache, dim=0)
    
    def _analyze_prototypes_from_logits(self, all_logits, fc_weight, epoch=0):
        """
        all_logits: [N, num_classes] from training_step (concatenated over the epoch)
        fc_weight: prototype matrix [num_classes, feature_dim]
        epoch: current epoch
        """
        num_samples, num_prototypes = all_logits.shape

        # Step 1: statistics
        probs = F.softmax(all_logits, dim=1)  # [N, num_classes]
        top1_proto = probs.argmax(dim=1)      # [N]

        # Usage count per prototype (top-1 assignment)
        usage_count = torch.bincount(top1_proto, minlength=num_prototypes)  # [num_classes]

        # Average max-softmax confidence when a prototype is selected
        max_probs, _ = probs.max(dim=1)
        avg_response = torch.zeros(num_prototypes, device=probs.device)
        for proto_idx in range(num_prototypes):
            selected = (top1_proto == proto_idx)
            if selected.any():
                avg_response[proto_idx] = max_probs[selected].mean()

        # Average assigned probability per prototype
        avg_proto_probs = probs.mean(dim=0)  # [num_classes]

        # Prototype cosine similarity matrix (normalized)
        proto_norm = F.normalize(fc_weight, dim=1)
        cosine_sim_matrix = torch.mm(proto_norm, proto_norm.t())  # [num_classes, num_classes]

        # Step 2: logging
        logger.info(f"=== [Epoch {epoch}] Prototype Analysis ===")
        logger.info(f"Usage Count: {usage_count.cpu().numpy()}")
        logger.info(f"Avg Response: {avg_response.cpu().numpy()}")
        logger.info(f"Avg Assigned Probability: {avg_proto_probs.cpu().numpy()}")
        logger.info(f"Avg Cosine Similarity: {cosine_sim_matrix.mean().item()}")

        return usage_count
    
    def _select_num_unlabeled_prototypes_to_keep(self, unlabeled_usage_counts, unlabeled_usage_coverage=0.9):
        """
        Decide how many unlabeled prototypes to keep based on cumulative usage coverage.

        Steps:
        1) Sort usage descending and find the smallest K that achieves `unlabeled_usage_coverage`
        2) Optionally extend K a bit if the tail is still "significant / unstable" (heuristic)

        Args:
            unlabeled_usage_counts (list[float] | np.ndarray):
                Usage count per *unlabeled* prototype (already excludes known prototypes).
            unlabeled_usage_coverage (float):
                Target cumulative coverage ratio, e.g., 0.90 / 0.99.

        Returns:
            int: number of unlabeled prototypes to keep (K).
        """
        unlabeled_usage_counts = np.asarray(unlabeled_usage_counts, dtype=np.float32)
        num_unlabeled = len(unlabeled_usage_counts)
        if num_unlabeled == 0:
            return 0

        sorted_usage = np.sort(unlabeled_usage_counts)[::-1]
        cum_usage = np.cumsum(sorted_usage)
        total_usage = float(cum_usage[-1])

        # Avoid division by zero when all counts are zero
        if total_usage <= 0:
            logger.warning("All unlabeled usage counts are zero; keep all unlabeled prototypes.")
            return num_unlabeled

        cum_coverage = cum_usage / total_usage

        # smallest index achieving coverage
        k_min = int(np.searchsorted(cum_coverage, unlabeled_usage_coverage))
        if k_min >= len(sorted_usage):
            logger.warning(f"{unlabeled_usage_coverage*100:.1f}% coverage reaches the end; keep all unlabeled prototypes.")
            return num_unlabeled

        tail_usage = sorted_usage[k_min:]
        if len(tail_usage) == 0:
            logger.warning(f"{unlabeled_usage_coverage*100:.1f}% coverage leaves no tail; keep K={k_min}.")
            return k_min

        tail_mean = float(np.mean(tail_usage))
        tail_delta = np.diff(tail_usage)
        tail_delta_mean = float(np.mean(np.abs(tail_delta))) if len(tail_delta) > 0 else 0.0

        k_final_idx = k_min
        for i in range(k_min, len(sorted_usage)):
            count_i = float(sorted_usage[i])
            delta_i = abs(float(tail_usage[i - k_min] - tail_usage[i - k_min - 1])) if (i - k_min - 1) >= 0 else 0.0

            if (count_i >= tail_mean) or (delta_i >= tail_delta_mean):
                k_final_idx = i
            else:
                break

        num_keep = k_final_idx + 1
        logger.info(
            f"[Keep-Unlabeled] coverage={unlabeled_usage_coverage:.3f}, "
            f"k_min={k_min}, tail_mean={tail_mean:.6f}, tail_delta_mean={tail_delta_mean:.6f}, "
            f"num_keep={num_keep}"
        )
        return num_keep


    def _merge_unlabeled_prototypes(self, fc_weight, usage_count, num_known_protos=9, unlabeled_usage_coverage=0.9544):
        """
        Merge low-usage unlabeled prototypes into a smaller set of high-usage anchors.

        Args:
            fc_weight (Tensor): prototype matrix [num_classes, feat_dim]
            usage_count (Tensor): usage count per prototype [num_classes]
            num_known_protos (int): number of labelled (known) prototypes to keep untouched (prefix part)
            unlabeled_usage_coverage (float): coverage threshold used to decide how many unlabeled prototypes to keep

        Returns:
            Tensor: merged prototype matrix [num_known_protos + K_temp, feat_dim]
        """
        usage_unlabeled = usage_count[num_known_protos:]
        proto_unlabeled = fc_weight[num_known_protos:].detach()

        usage_list = usage_unlabeled.tolist()

        K_temp = self._select_num_unlabeled_prototypes_to_keep(
            usage_list,
            unlabeled_usage_coverage=unlabeled_usage_coverage
        )

        logger.info(f"Auto selected {K_temp} unlabeled prototypes to keep.")

        # Step 1: sort unlabeled prototypes by usage (descending)
        sorted_counts, sorted_indices = torch.sort(usage_unlabeled, descending=True)
        high_resp_protos = proto_unlabeled[sorted_indices[:K_temp]].clone()
        low_resp_protos = proto_unlabeled[sorted_indices[K_temp:]]

        # Step 2: compute similarity between low-usage protos and anchors
        sims_matrix = F.cosine_similarity(
            low_resp_protos.unsqueeze(1),    # [num_low, 1, dim]
            high_resp_protos.unsqueeze(0),   # [1, num_high, dim]
            dim=-1                           # -> [num_low, num_high]
        )
        closest_indices = sims_matrix.argmax(dim=1)  # [num_low]

        # Step 3: merge assigned low-usage protos into each anchor
        for idx in range(K_temp):
            assigned_low_proto = low_resp_protos[closest_indices == idx]
            if assigned_low_proto.shape[0] > 0:
                merged_proto = torch.cat([
                    high_resp_protos[idx].unsqueeze(0),
                    assigned_low_proto
                ], dim=0).mean(dim=0)
                high_resp_protos[idx] = F.normalize(merged_proto, dim=0)

        # Step 4: concatenate labelled prototypes + merged unlabeled anchors
        final_fc2_weight = torch.cat([
            fc_weight[:num_known_protos].detach(),
            high_resp_protos
        ], dim=0)

        return final_fc2_weight
