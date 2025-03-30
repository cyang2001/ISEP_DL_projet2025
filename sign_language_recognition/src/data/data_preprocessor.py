"""
Module de prétraitement des données vidéo pour la reconnaissance de langue des signes.
Adapté du projet WLASL: https://github.com/dxli94/WLASL
"""
import os
import cv2
import json
import numpy as np
from omegaconf import DictConfig
import torch
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union

from src.utils.logger import get_logger


class VideoPreprocessor:
    """
    Classe pour prétraiter les vidéos pour la reconnaissance de langue des signes.
    Adapté pour le traitement du dataset WLASL.
    """
    
    def __init__(self, cfg: DictConfig, logger=None):
        """
        Initialise le préprocesseur vidéo.
        
        Args:
            cfg (DictConfig): Configuration de prétraitement.
            logger (Logger, optional): Logger pour la journalisation.
        """
        self.cfg = cfg
        self.m_cfg = cfg.model
        self.logger = logger or get_logger(__name__)
        
        # Paramètres de prétraitement vidéo
        video_cfg = self.m_cfg.data_preprocessing.video
        self.target_height = video_cfg.target_height
        self.target_width = video_cfg.target_width
        self.target_fps = video_cfg.target_fps
        self.max_frames = video_cfg.max_frames
        self.min_frames = video_cfg.min_frames
        self.normalize = video_cfg.normalize
        self.crop_method = video_cfg.crop_method
        
        # Créer les répertoires nécessaires
        os.makedirs(self.cfg.data.processed_dir, exist_ok=True)
        
        # Liste pour stocker les fichiers manquants
        self.missing_videos = []
        
        self.logger.info(f"Initialisé VideoPreprocessor pour WLASL avec target_height={self.target_height}, "
                        f"target_width={self.target_width}, target_fps={self.target_fps}")
    
    def load_wlasl_json(self, json_path: str) -> List[Dict]:
        """
        Charge les annotations du dataset WLASL.
        
        Args:
            json_path (str): Chemin vers le fichier JSON d'annotations WLASL.
            
        Returns:
            List[Dict]: Liste des entrées du dataset.
        """
        self.logger.info(f"Chargement des annotations WLASL depuis {json_path}")
        
        if not os.path.exists(json_path):
            self.logger.error(f"Fichier JSON introuvable: {json_path}")
            raise FileNotFoundError(f"Fichier JSON introuvable: {json_path}")
        
        with open(json_path, 'r') as f:
            wlasl_data = json.load(f)
        
        self.logger.info(f"Chargé {len(wlasl_data)} classes depuis les annotations WLASL")
        return wlasl_data
    
    def extract_video_segment(self, video_path: str, frame_start: int, frame_end: int) -> np.ndarray:
        """
        Extrait un segment spécifique d'une vidéo entre les frames indiquées.
        
        Args:
            video_path (str): Chemin vers le fichier vidéo.
            frame_start (int): Frame de début (indexé à partir de 1).
            frame_end (int): Frame de fin (-1 pour la dernière frame).
            
        Returns:
            np.ndarray: Segment vidéo extrait de forme [num_frames, H, W, C].
        """
        if not os.path.exists(video_path):
            self.logger.error(f"Fichier vidéo introuvable: {video_path}")
            raise FileNotFoundError(f"Fichier vidéo introuvable: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.logger.error(f"Impossible d'ouvrir la vidéo: {video_path}")
            raise ValueError(f"Impossible d'ouvrir la vidéo: {video_path}")
        
        # Convertir l'indexation à partir de 1 (WLASL) à l'indexation à partir de 0
        frame_start = max(0, frame_start - 1)
        
        # Obtenir le nombre total de frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Si frame_end est -1, utiliser la dernière frame
        if frame_end == -1:
            frame_end = total_frames - 1
        else:
            frame_end = min(frame_end - 1, total_frames - 1)
        
        if frame_start >= frame_end:
            self.logger.warning(f"Plage de frames invalide: {frame_start+1} à {frame_end+1} pour {video_path}")
            # Utiliser toute la vidéo si la plage est invalide
            frame_start = 0
            frame_end = total_frames - 1
        
        # Nombre de frames à extraire
        num_frames = frame_end - frame_start + 1
        
        # Avancer à la première frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
        
        frames = []
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        if not frames:
            self.logger.error(f"Aucune frame extraite de {video_path}")
            raise ValueError(f"Aucune frame extraite de {video_path}")
        
        self.logger.debug(f"Extrait {len(frames)} frames de {video_path} (frames {frame_start+1} à {frame_end+1})")
        return np.array(frames)
    
    def preprocess_video(self, video_path: str, frame_start: Optional[int] = None, 
                         frame_end: Optional[int] = None) -> np.ndarray:
        """
        Prétraite une vidéo en extrayant et redimensionnant les cadres.
        
        Args:
            video_path (str): Chemin d'accès au fichier vidéo.
            frame_start (int, optional): Frame de début (indexé à partir de 1).
            frame_end (int, optional): Frame de fin (-1 pour la dernière frame).
            
        Returns:
            np.ndarray: Tableau de cadres vidéo prétraités de forme [num_frames, H, W, C].
        """
        self.logger.debug(f"Prétraitement de la vidéo: {video_path}")
        
        # Extraire le segment vidéo si des limites sont fournies
        if frame_start is not None and frame_end is not None:
            frames = self.extract_video_segment(video_path, frame_start, frame_end)
        else:
            # Sinon, extraire toute la vidéo
            if not os.path.exists(video_path):
                self.logger.error(f"Fichier vidéo introuvable: {video_path}")
                raise FileNotFoundError(f"Fichier vidéo introuvable: {video_path}")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.logger.error(f"Impossible d'ouvrir la vidéo: {video_path}")
                raise ValueError(f"Impossible d'ouvrir la vidéo: {video_path}")
            
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            cap.release()
            
            if not frames:
                self.logger.error(f"Aucune frame extraite de {video_path}")
                raise ValueError(f"Aucune frame extraite de {video_path}")
            
            frames = np.array(frames)
        
        # Ajuster le nombre de frames
        num_frames = len(frames)
        
        # Respecter le nombre minimum de frames
        if num_frames < self.min_frames:
            # Dupliquer les frames pour atteindre le minimum
            indices = np.linspace(0, num_frames - 1, self.min_frames, dtype=int)
            frames = frames[indices]
        
        # Respecter le nombre maximum de frames
        if len(frames) > self.max_frames:
            # Échantillonner uniformément
            indices = np.linspace(0, len(frames) - 1, self.max_frames, dtype=int)
            frames = frames[indices]
        
        # Appliquer le recadrage si nécessaire
        if self.crop_method != "none":
            frames = self._apply_crop(frames)
        
        # Redimensionner toutes les frames
        resized_frames = []
        for frame in frames:
            resized = cv2.resize(frame, (self.target_width, self.target_height))
            resized_frames.append(resized)
        
        frames = np.array(resized_frames)
        
        # Normaliser si nécessaire
        if self.normalize:
            frames = frames.astype(np.float32) / 255.0
        
        self.logger.debug(f"Prétraitement terminé pour {video_path}, forme: {frames.shape}")
        return frames
    
    def _apply_crop(self, frames: np.ndarray) -> np.ndarray:
        """
        Applique le recadrage aux frames selon la méthode spécifiée.
        
        Args:
            frames (np.ndarray): Frames à recadrer.
            
        Returns:
            np.ndarray: Frames recadrées.
        """
        h, w = frames[0].shape[:2]
        
        if self.crop_method == "center":
            # Recadrage central
            new_size = min(h, w)
            start_x = (w - new_size) // 2
            start_y = (h - new_size) // 2
            
            cropped_frames = []
            for frame in frames:
                cropped = frame[start_y:start_y+new_size, start_x:start_x+new_size]
                cropped_frames.append(cropped)
            
            return np.array(cropped_frames)
        
        elif self.crop_method == "random":
            # Recadrage aléatoire (même recadrage pour toutes les frames d'une vidéo)
            new_size = min(h, w)
            max_x = w - new_size
            max_y = h - new_size
            
            if max_x > 0:
                start_x = np.random.randint(0, max_x)
            else:
                start_x = 0
                
            if max_y > 0:
                start_y = np.random.randint(0, max_y)
            else:
                start_y = 0
            
            cropped_frames = []
            for frame in frames:
                cropped = frame[start_y:start_y+new_size, start_x:start_x+new_size]
                cropped_frames.append(cropped)
            
            return np.array(cropped_frames)
        
        # Si la méthode n'est pas reconnue ou est "none", retourner les frames inchangées
        return frames
    
    def _find_video_file(self, video_dir: str, video_id: str) -> Optional[str]:
        """
        Tente de trouver un fichier vidéo dans différents formats.
        
        Args:
            video_dir (str): Répertoire contenant les vidéos.
            video_id (str): ID de la vidéo à rechercher.
            
        Returns:
            Optional[str]: Chemin vers le fichier vidéo trouvé, ou None si non trouvé.
        """
        # Extensions vidéo connues
        extensions = ['.mp4', '.avi', '.mov', '.webm']
        
        for ext in extensions:
            video_path = os.path.join(video_dir, f"{video_id}{ext}")
            if os.path.exists(video_path):
                return video_path
        
        return None
    
    def process_wlasl_video(self, video_path: str, gloss: str, instance_id: int, 
                           frame_start: Optional[int] = None, frame_end: Optional[int] = None,
                           output_dir: Optional[str] = None) -> str:
        """
        Traite une vidéo WLASL et sauvegarde le résultat.
        
        Args:
            video_path (str): Chemin d'accès au fichier vidéo.
            gloss (str): Étiquette du geste.
            instance_id (int): ID de l'instance.
            frame_start (int, optional): Frame de début (indexé à partir de 1).
            frame_end (int, optional): Frame de fin (-1 pour la dernière frame).
            output_dir (str, optional): Répertoire de sortie.
            
        Returns:
            str: Chemin du fichier traité.
        """
        output_dir = output_dir or self.cfg.data.processed_dir
        
        # Créer le répertoire par classe si nécessaire
        class_dir = os.path.join(output_dir, gloss)
        os.makedirs(class_dir, exist_ok=True)
        
        try:
            # Prétraiter la vidéo
            frames = self.preprocess_video(video_path, frame_start, frame_end)
            
            # Nom du fichier de sortie basé sur le gloss et l'ID de l'instance
            output_filename = f"{gloss}_{instance_id}.npy"
            output_path = os.path.join(class_dir, output_filename)
            
            # Sauvegarder les frames prétraitées
            np.save(output_path, frames)
            
            self.logger.debug(f"Sauvegardé {video_path} vers {output_path}")
            return output_path
        
        except Exception as e:
            self.logger.error(f"Erreur lors du traitement de {video_path}: {e}")
            raise
    
    def process_wlasl_dataset(self, wlasl_json_path: str, video_dir: str, output_dir: Optional[str] = None) -> Dict[str, List[str]]:
        """
        Traite l'ensemble du dataset WLASL.
        
        Args:
            wlasl_json_path (str): Chemin d'accès au fichier JSON des annotations WLASL.
            video_dir (str): Répertoire contenant les vidéos.
            output_dir (str, optional): Répertoire de sortie.
            
        Returns:
            Dict[str, List[str]]: Dictionnaire des fichiers traités par classe.
        """
        output_dir = output_dir or self.cfg.data.processed_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.logger.info(f"Traitement du dataset WLASL depuis {wlasl_json_path}")
        
        # Charger les annotations
        wlasl_data = self.load_wlasl_json(wlasl_json_path)
        
        # Réinitialiser la liste des vidéos manquantes
        self.missing_videos = []
        
        results = {}
        total_processed = 0
        total_videos = 0
        
        # Pour chaque classe (gloss)
        for entry in tqdm(wlasl_data, desc="Traitement des classes"):
            gloss = entry["gloss"]
            results[gloss] = []
            
            # Pour chaque instance de cette classe
            for instance in entry["instances"]:
                total_videos += 1
                video_id = instance["video_id"]
                frame_start = instance.get("frame_start")
                frame_end = instance.get("frame_end")
                instance_id = instance["instance_id"]
                
                # Tenter de trouver le fichier vidéo
                video_path = self._find_video_file(video_dir, video_id)
                
                if not video_path:
                    self.logger.warning(f"Vidéo introuvable pour {video_id}, gloss={gloss}")
                    # Enregistrer cette vidéo comme manquante
                    self.missing_videos.append({
                        "video_id": video_id,
                        "gloss": gloss,
                        "instance_id": instance_id
                    })
                    continue
                
                try:
                    # Traiter la vidéo
                    output_path = self.process_wlasl_video(
                        video_path, gloss, instance_id, frame_start, frame_end, output_dir
                    )
                    results[gloss].append(output_path)
                    total_processed += 1
                except Exception as e:
                    self.logger.error(f"Erreur lors du traitement de {video_id}: {e}")
        
        self.logger.info(f"Traitement terminé. {total_processed}/{total_videos} vidéos traitées avec succès.")
        self.logger.info(f"Vidéos manquantes: {len(self.missing_videos)}/{total_videos}")
        
        return results
    
    def save_missing_videos_list(self, output_path: Optional[str] = None) -> str:
        """
        Sauvegarde la liste des vidéos manquantes dans un fichier texte.
        
        Args:
            output_path (str, optional): Chemin du fichier de sortie.
            
        Returns:
            str: Chemin du fichier créé.
        """
        if not self.missing_videos:
            self.logger.info("Aucune vidéo manquante à enregistrer.")
            return ""
        
        output_path = output_path or self.cfg.data.missing_videos_log
        
        try:
            with open(output_path, 'w') as f:
                f.write("video_id,gloss,instance_id\n")
                for video in self.missing_videos:
                    f.write(f"{video['video_id']},{video['gloss']},{video['instance_id']}\n")
            
            self.logger.info(f"Liste des {len(self.missing_videos)} vidéos manquantes sauvegardée dans {output_path}")
            return output_path
        
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde de la liste des vidéos manquantes: {e}")
            return ""
    
    def process_dataset(self, video_dir: str, output_dir: Optional[str] = None, 
                       wlasl_json_path: Optional[str] = None) -> Dict:
        """
        Traite tous les fichiers vidéo dans un répertoire.
        Si wlasl_json_path est fourni, utilise le format WLASL pour le traitement.
        
        Args:
            video_dir (str): Répertoire contenant les vidéos à traiter.
            output_dir (str, optional): Répertoire de sortie pour les vidéos traitées.
            wlasl_json_path (str, optional): Chemin d'accès au fichier JSON des annotations WLASL.
                
        Returns:
            dict: Dictionnaire de mappages entre les noms de fichiers et les chemins de sortie.
        """
        output_dir = output_dir or self.cfg.data.processed_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Si le chemin JSON WLASL est fourni, traiter en tant que dataset WLASL
        if wlasl_json_path:
            result = self.process_wlasl_dataset(wlasl_json_path, video_dir, output_dir)
            # Sauvegarder la liste des vidéos manquantes
            self.save_missing_videos_list()
            return result
        
        # Sinon, traiter comme un répertoire général de vidéos
        self.logger.info(f"Traitement des vidéos de {video_dir} vers {output_dir}")
        
        video_files = [f for f in os.listdir(video_dir) 
                      if f.endswith(('.mp4', '.avi', '.mov', '.webm'))]
        
        results = {}
        for video_file in tqdm(video_files, desc="Traitement des vidéos"):
            video_path = os.path.join(video_dir, video_file)
            try:
                # Prétraiter la vidéo
                frames = self.preprocess_video(video_path)
                
                # Créer un nom de fichier de sortie
                base_name = os.path.splitext(video_file)[0]
                output_path = os.path.join(output_dir, f"{base_name}.npy")
                
                # Sauvegarder les frames
                np.save(output_path, frames)
                
                results[video_file] = output_path
                self.logger.debug(f"Sauvegardé {video_file} vers {output_path}")
            except Exception as e:
                self.logger.error(f"Erreur lors du traitement de {video_file}: {e}")
        
        self.logger.info(f"Traitement terminé. {len(results)} vidéos traitées avec succès.")
        return results 