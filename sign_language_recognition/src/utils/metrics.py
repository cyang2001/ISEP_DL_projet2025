"""
Module de métriques d'évaluation pour la reconnaissance de langue des signes.
"""
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import wandb


class EvaluationMetrics:
    """
    Classe pour calculer et visualiser les métriques d'évaluation.
    """
    
    def __init__(self, class_names=None):
        """
        Initialise la classe EvaluationMetrics.
        
        Args:
            class_names (list, optional): Liste des noms de classes pour la visualisation.
        """
        self.class_names = class_names
        self.reset()
    
    def reset(self):
        """
        Réinitialise toutes les métriques.
        """
        self.all_preds = []
        self.all_targets = []
    
    def update(self, preds, targets):
        """
        Met à jour les métriques avec de nouvelles prédictions et cibles.
        
        Args:
            preds (np.ndarray or torch.Tensor): Prédictions du modèle.
            targets (np.ndarray or torch.Tensor): Valeurs cibles réelles.
        """
        # Convertir en numpy si nécessaire
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()
            
        # Pour les prédictions à plusieurs classes, obtenir l'indice de la classe max
        if len(preds.shape) > 1 and preds.shape[1] > 1:
            preds = np.argmax(preds, axis=1)
            
        self.all_preds.extend(preds)
        self.all_targets.extend(targets)
    
    def compute_metrics(self):
        """
        Calcule toutes les métriques d'évaluation.
        
        Returns:
            dict: Dictionnaire contenant toutes les métriques calculées.
        """
        preds = np.array(self.all_preds)
        targets = np.array(self.all_targets)
        
        metrics = {
            'accuracy': accuracy_score(targets, preds),
            'precision': precision_score(targets, preds, average='macro', zero_division=0),
            'recall': recall_score(targets, preds, average='macro', zero_division=0),
            'f1': f1_score(targets, preds, average='macro', zero_division=0)
        }
        
        # Calcul de la matrice de confusion
        metrics['confusion_matrix'] = confusion_matrix(targets, preds)
        
        return metrics
    
    def log_metrics(self, metrics, prefix=''):
        """
        Enregistre les métriques dans Weights & Biases.
        
        Args:
            metrics (dict): Dictionnaire des métriques à enregistrer.
            prefix (str, optional): Préfixe pour les noms des métriques.
        """
        if wandb.run is None:
            return
            
        log_dict = {}
        for metric_name, metric_value in metrics.items():
            if metric_name == 'confusion_matrix':
                continue
            log_dict[f"{prefix}{metric_name}"] = metric_value
            
        # Enregistrer la matrice de confusion comme image
        if 'confusion_matrix' in metrics:
            cm = metrics['confusion_matrix']
            fig = self.plot_confusion_matrix(cm)
            log_dict[f"{prefix}confusion_matrix"] = wandb.Image(fig)
            plt.close(fig)
            
        wandb.log(log_dict)
    
    def plot_confusion_matrix(self, cm, figsize=(10, 8)):
        """
        Génère une visualisation de la matrice de confusion.
        
        Args:
            cm (np.ndarray): Matrice de confusion.
            figsize (tuple, optional): Taille de la figure.
            
        Returns:
            matplotlib.figure.Figure: Figure matplotlib de la matrice de confusion.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if self.class_names is not None:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.class_names,
                       yticklabels=self.class_names)
        else:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            
        plt.ylabel('Valeur Réelle')
        plt.xlabel('Prédiction')
        plt.title('Matrice de Confusion')
        
        return fig
    
    def get_summary(self):
        """
        Renvoie un résumé des métriques sous forme de chaîne de caractères.
        
        Returns:
            str: Résumé formaté des métriques.
        """
        metrics = self.compute_metrics()
        
        summary = (
            f"Accuracy: {metrics['accuracy']:.4f}\n"
            f"Precision: {metrics['precision']:.4f}\n"
            f"Recall: {metrics['recall']:.4f}\n"
            f"F1 Score: {metrics['f1']:.4f}\n"
        )
        
        return summary


class SequenceEvaluationMetrics(EvaluationMetrics):
    """
    Classe pour l'évaluation des métriques de séquence (pour la reconnaissance continue).
    """
    
    def compute_edit_distance(self, pred_sequence, target_sequence):
        """
        Calcule la distance d'édition entre deux séquences.
        
        Args:
            pred_sequence (list): Séquence prédite.
            target_sequence (list): Séquence cible.
            
        Returns:
            float: Distance d'édition normalisée.
        """
        m, n = len(pred_sequence), len(target_sequence)
        
        # Initialiser la matrice de distance
        dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
        
        # Cas de base
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
            
        # Remplir la matrice de programmation dynamique
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred_sequence[i-1] == target_sequence[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j],      # Suppression
                                      dp[i][j-1],      # Insertion
                                      dp[i-1][j-1])    # Substitution
                                      
        # Normaliser par la longueur de la séquence la plus longue
        max_len = max(m, n)
        if max_len == 0:
            return 0.0
        
        return dp[m][n] / max_len
    
    def update_sequences(self, pred_sequences, target_sequences):
        """
        Met à jour les métriques avec de nouvelles séquences de prédiction et cibles.
        
        Args:
            pred_sequences (list): Liste des séquences prédites.
            target_sequences (list): Liste des séquences cibles.
        """
        self.sequence_preds = pred_sequences
        self.sequence_targets = target_sequences
    
    def compute_sequence_metrics(self):
        """
        Calcule les métriques pour les séquences.
        
        Returns:
            dict: Dictionnaire contenant les métriques de séquence.
        """
        if not hasattr(self, 'sequence_preds') or not hasattr(self, 'sequence_targets'):
            return {}
            
        edit_distances = []
        
        for pred_seq, target_seq in zip(self.sequence_preds, self.sequence_targets):
            edit_dist = self.compute_edit_distance(pred_seq, target_seq)
            edit_distances.append(edit_dist)
            
        mean_edit_distance = np.mean(edit_distances)
        
        return {
            'mean_edit_distance': mean_edit_distance
        } 