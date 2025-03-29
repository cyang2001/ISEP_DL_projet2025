"""
Module de visualisation pour le projet de reconnaissance de langue des signes.
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import os
from matplotlib.animation import FuncAnimation
import wandb


class VideoVisualizer:
    """
    Classe pour visualiser les vidéos et les résultats de prédiction.
    """
    
    def __init__(self):
        """
        Initialise la classe VideoVisualizer.
        """
        pass
    
    def plot_video_frames(self, video_frames, num_frames=5, figsize=(15, 3)):
        """
        Affiche un échantillon des cadres d'une vidéo.
        
        Args:
            video_frames (np.ndarray): Tableau de forme [T, H, W, C] contenant les cadres de la vidéo.
            num_frames (int, optional): Nombre de cadres à afficher.
            figsize (tuple, optional): Taille de la figure.
            
        Returns:
            matplotlib.figure.Figure: La figure créée.
        """
        if isinstance(video_frames, torch.Tensor):
            video_frames = video_frames.detach().cpu().numpy()
            
        # Normaliser si nécessaire
        if video_frames.max() <= 1.0:
            video_frames = (video_frames * 255).astype(np.uint8)
            
        # Sélectionner des cadres uniformément espacés
        total_frames = video_frames.shape[0]
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        # Créer la figure
        fig, axes = plt.subplots(1, num_frames, figsize=figsize)
        
        for i, idx in enumerate(indices):
            if num_frames == 1:
                ax = axes
            else:
                ax = axes[i]
            
            frame = video_frames[idx]
            
            # Convertir BGR en RGB si nécessaire
            if frame.shape[-1] == 3:  # Si c'est une image couleur
                if np.mean(frame[:, :, 0]) > np.mean(frame[:, :, 2]):  # Heuristique simple pour détecter BGR
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            ax.imshow(frame)
            ax.set_title(f"Cadre {idx}")
            ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def create_video_animation(self, video_frames, output_path=None, fps=10):
        """
        Crée une animation à partir d'une séquence de cadres vidéo.
        
        Args:
            video_frames (np.ndarray): Tableau de forme [T, H, W, C] contenant les cadres de la vidéo.
            output_path (str, optional): Chemin de sortie pour enregistrer l'animation.
            fps (int, optional): Images par seconde pour l'animation.
            
        Returns:
            matplotlib.animation.FuncAnimation: Objet d'animation.
        """
        if isinstance(video_frames, torch.Tensor):
            video_frames = video_frames.detach().cpu().numpy()
            
        # Normaliser si nécessaire
        if video_frames.max() <= 1.0:
            video_frames = (video_frames * 255).astype(np.uint8)
            
        # Convertir BGR en RGB si nécessaire
        for i in range(len(video_frames)):
            if video_frames.shape[-1] == 3:  # Si c'est une image couleur
                if np.mean(video_frames[i, :, :, 0]) > np.mean(video_frames[i, :, :, 2]):
                    video_frames[i] = cv2.cvtColor(video_frames[i], cv2.COLOR_BGR2RGB)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.axis('off')
        img = ax.imshow(video_frames[0])
        
        def update(frame):
            img.set_array(video_frames[frame])
            return [img]
        
        ani = FuncAnimation(fig, update, frames=len(video_frames), 
                            interval=1000/fps, blit=True)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            ani.save(output_path, writer='pillow', fps=fps)
        
        return ani
    
    def visualize_with_landmarks(self, video_frame, landmarks, connections=None, 
                                radius=3, thickness=2, landmark_color=(0, 255, 0),
                                connection_color=(0, 0, 255)):
        """
        Visualise une image avec des points de repère et leurs connexions.
        
        Args:
            video_frame (np.ndarray): Image à visualiser.
            landmarks (np.ndarray): Points de repère de forme [N, 2] ou [N, 3].
            connections (list, optional): Liste des paires d'indices pour dessiner des connexions.
            radius (int, optional): Rayon des points de repère.
            thickness (int, optional): Épaisseur des lignes de connexion.
            landmark_color (tuple, optional): Couleur des points de repère (BGR).
            connection_color (tuple, optional): Couleur des connexions (BGR).
            
        Returns:
            np.ndarray: Image avec les points de repère visualisés.
        """
        if isinstance(video_frame, torch.Tensor):
            video_frame = video_frame.detach().cpu().numpy()
        
        if isinstance(landmarks, torch.Tensor):
            landmarks = landmarks.detach().cpu().numpy()
            
        # Normaliser si nécessaire
        if video_frame.max() <= 1.0:
            video_frame = (video_frame * 255).astype(np.uint8)
            
        # Créer une copie pour ne pas modifier l'original
        vis_img = video_frame.copy()
        
        # Dessiner les points de repère
        for lm in landmarks:
            x, y = int(lm[0]), int(lm[1])
            cv2.circle(vis_img, (x, y), radius, landmark_color, -1)
        
        # Dessiner les connexions si fournies
        if connections:
            for connection in connections:
                idx1, idx2 = connection
                if idx1 < len(landmarks) and idx2 < len(landmarks):
                    pt1 = (int(landmarks[idx1][0]), int(landmarks[idx1][1]))
                    pt2 = (int(landmarks[idx2][0]), int(landmarks[idx2][1]))
                    cv2.line(vis_img, pt1, pt2, connection_color, thickness)
        
        return vis_img
    
    def log_video_to_wandb(self, video_frames, caption="Video"):
        """
        Enregistre une vidéo dans Weights & Biases.
        
        Args:
            video_frames (np.ndarray): Tableau de forme [T, H, W, C] contenant les cadres de la vidéo.
            caption (str, optional): Légende pour la vidéo.
        """
        if wandb.run is None:
            return
            
        if isinstance(video_frames, torch.Tensor):
            video_frames = video_frames.detach().cpu().numpy()
            
        # Normaliser si nécessaire
        if video_frames.max() <= 1.0:
            video_frames = (video_frames * 255).astype(np.uint8)
            
        # Convertir en format uint8
        video_frames = video_frames.astype(np.uint8)
        
        # Enregistrer la vidéo temporairement
        temp_path = "temp_video.mp4"
        
        # Créer un objet VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        h, w = video_frames.shape[1:3]
        out = cv2.VideoWriter(temp_path, fourcc, 10, (w, h))
        
        # Écrire les cadres
        for frame in video_frames:
            out.write(frame if frame.shape[-1] == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
        
        out.release()
        
        # Enregistrer dans wandb
        wandb.log({caption: wandb.Video(temp_path, fps=10, format="mp4")})
        
        # Supprimer le fichier temporaire
        if os.path.exists(temp_path):
            os.remove(temp_path)


class SkeletonVisualizer:
    """
    Classe pour visualiser les données de squelette.
    """
    
    def __init__(self):
        """
        Initialise la classe SkeletonVisualizer.
        """
        # Définir les connexions standard pour les mains MediaPipe
        self.mediapipe_hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Pouce
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Majeur
            (0, 13), (13, 14), (14, 15), (15, 16),  # Annulaire
            (0, 17), (17, 18), (18, 19), (19, 20)  # Auriculaire
        ]
    
    def plot_skeleton_2d(self, skeleton_points, connections=None, figsize=(8, 8)):
        """
        Visualise un squelette 2D.
        
        Args:
            skeleton_points (np.ndarray): Points du squelette de forme [N, 2].
            connections (list, optional): Liste des paires d'indices pour dessiner des connexions.
            figsize (tuple, optional): Taille de la figure.
            
        Returns:
            matplotlib.figure.Figure: La figure créée.
        """
        if isinstance(skeleton_points, torch.Tensor):
            skeleton_points = skeleton_points.detach().cpu().numpy()
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Tracer les points
        ax.scatter(skeleton_points[:, 0], skeleton_points[:, 1], c='blue', s=50)
        
        # Tracer les connexions
        if connections is None:
            connections = self.mediapipe_hand_connections
            
        for connection in connections:
            idx1, idx2 = connection
            if idx1 < len(skeleton_points) and idx2 < len(skeleton_points):
                ax.plot([skeleton_points[idx1, 0], skeleton_points[idx2, 0]],
                        [skeleton_points[idx1, 1], skeleton_points[idx2, 1]], 'r-')
        
        # Inverser l'axe y pour correspondre aux coordonnées de l'image
        ax.invert_yaxis()
        
        # Ajouter des labels pour les points
        for i, (x, y) in enumerate(skeleton_points):
            ax.annotate(str(i), (x, y), textcoords="offset points", 
                        xytext=(0, 10), ha='center')
        
        plt.title('Visualisation du Squelette 2D')
        plt.grid(True)
        plt.tight_layout()
        
        return fig
    
    def plot_skeleton_sequence(self, skeleton_sequence, num_frames=5, connections=None, figsize=(15, 3)):
        """
        Visualise une séquence de squelettes.
        
        Args:
            skeleton_sequence (np.ndarray): Séquence de squelettes de forme [T, N, 2/3].
            num_frames (int, optional): Nombre de cadres à afficher.
            connections (list, optional): Liste des paires d'indices pour dessiner des connexions.
            figsize (tuple, optional): Taille de la figure.
            
        Returns:
            matplotlib.figure.Figure: La figure créée.
        """
        if isinstance(skeleton_sequence, torch.Tensor):
            skeleton_sequence = skeleton_sequence.detach().cpu().numpy()
            
        # Sélectionner des cadres uniformément espacés
        total_frames = skeleton_sequence.shape[0]
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        # Créer la figure
        fig, axes = plt.subplots(1, num_frames, figsize=figsize)
        
        if connections is None:
            connections = self.mediapipe_hand_connections
            
        for i, idx in enumerate(indices):
            if num_frames == 1:
                ax = axes
            else:
                ax = axes[i]
            
            skeleton = skeleton_sequence[idx]
            
            # Tracer les points
            ax.scatter(skeleton[:, 0], skeleton[:, 1], c='blue', s=20)
            
            # Tracer les connexions
            for connection in connections:
                idx1, idx2 = connection
                if idx1 < len(skeleton) and idx2 < len(skeleton):
                    ax.plot([skeleton[idx1, 0], skeleton[idx2, 0]],
                            [skeleton[idx1, 1], skeleton[idx2, 1]], 'r-')
            
            # Inverser l'axe y pour correspondre aux coordonnées de l'image
            ax.invert_yaxis()
            ax.set_title(f"Cadre {idx}")
            ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def overlay_skeleton_on_video(self, video_frames, skeleton_sequence, connections=None):
        """
        Superpose une séquence de squelettes sur les cadres vidéo.
        
        Args:
            video_frames (np.ndarray): Tableau de forme [T, H, W, C] contenant les cadres de la vidéo.
            skeleton_sequence (np.ndarray): Séquence de squelettes de forme [T, N, 2/3].
            connections (list, optional): Liste des paires d'indices pour dessiner des connexions.
            
        Returns:
            np.ndarray: Vidéo avec les squelettes superposés.
        """
        if isinstance(video_frames, torch.Tensor):
            video_frames = video_frames.detach().cpu().numpy()
            
        if isinstance(skeleton_sequence, torch.Tensor):
            skeleton_sequence = skeleton_sequence.detach().cpu().numpy()
            
        # Normaliser les cadres vidéo si nécessaire
        if video_frames.max() <= 1.0:
            video_frames = (video_frames * 255).astype(np.uint8)
            
        # S'assurer que les deux séquences ont la même longueur
        min_len = min(len(video_frames), len(skeleton_sequence))
        video_frames = video_frames[:min_len]
        skeleton_sequence = skeleton_sequence[:min_len]
        
        # Créer une copie pour ne pas modifier l'original
        overlaid_frames = video_frames.copy()
        
        if connections is None:
            connections = self.mediapipe_hand_connections
            
        # Superposer les squelettes sur chaque cadre
        for i in range(min_len):
            frame = overlaid_frames[i]
            skeleton = skeleton_sequence[i]
            
            # Dessiner les points
            for j, point in enumerate(skeleton):
                x, y = int(point[0]), int(point[1])
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                
            # Dessiner les connexions
            for connection in connections:
                idx1, idx2 = connection
                if idx1 < len(skeleton) and idx2 < len(skeleton):
                    pt1 = (int(skeleton[idx1][0]), int(skeleton[idx1][1]))
                    pt2 = (int(skeleton[idx2][0]), int(skeleton[idx2][1]))
                    cv2.line(frame, pt1, pt2, (0, 0, 255), 2)
            
            overlaid_frames[i] = frame
        
        return overlaid_frames 