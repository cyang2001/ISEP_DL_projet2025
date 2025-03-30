"""
Module principal pour le prétraitement des données.
Adapté pour le traitement du dataset WLASL: https://github.com/dxli94/WLASL
"""
import logging
import os
import argparse
from omegaconf import DictConfig

from src.utils.logger import get_logger
from src.data.data_preprocessor import VideoPreprocessor


def main(cfg: DictConfig, logger: logging.Logger=None):
    """
    Fonction principale pour prétraiter les données vidéo.
    Supporte le format de données WLASL.
    
    Args:
        cfg (DictConfig): Configuration hydra.
    """
    logger = get_logger(__name__)
    logger.info("Démarrage du prétraitement des données vidéo...")
    
    preprocessor = VideoPreprocessor(cfg, logger)
    raw_dir = cfg.data.raw_dir
    processed_dir = cfg.data.processed_dir
    
    if not os.path.exists(raw_dir):
        logger.error(f"Répertoire de données brutes introuvable: {raw_dir}")
        return
    
    # Vérifier si un fichier JSON WLASL est spécifié
    wlasl_json_path = cfg.data.wlasl_json_path
    if wlasl_json_path:
        if os.path.exists(wlasl_json_path):
            logger.info(f"Utilisation du fichier d'annotations WLASL: {wlasl_json_path}")
            # Traiter le dataset WLASL
            result = preprocessor.process_dataset(raw_dir, processed_dir, wlasl_json_path)
            
            # Une fois le traitement terminé, vérifier les vidéos manquantes
            missing_log_path = cfg.data.missing_videos_log
            
            # Créer une liste des classes (pour le mode WLASL)
            if result:
                classes_file = os.path.join(processed_dir, "classes.txt")
                with open(classes_file, 'w') as f:
                    for class_name in sorted(result.keys()):
                        f.write(f"{class_name}\n")
                logger.info(f"Liste des classes sauvegardée dans {classes_file}")
                
                # Générer le rapport de statistiques
                total_videos = sum(len(videos) for videos in result.values())
                total_classes = len(result)
                stats_file = os.path.join(processed_dir, "dataset_stats.txt")
                with open(stats_file, 'w') as f:
                    f.write(f"WLASL Dataset Statistics\n")
                    f.write(f"----------------------\n")
                    f.write(f"Total classes: {total_classes}\n")
                    f.write(f"Total videos processed: {total_videos}\n")
                    f.write(f"\nTop 10 classes by number of samples:\n")
                    
                    # Trier les classes par nombre d'échantillons
                    sorted_classes = sorted(result.items(), key=lambda x: len(x[1]), reverse=True)
                    for i, (class_name, videos) in enumerate(sorted_classes[:10]):
                        f.write(f"{i+1}. {class_name}: {len(videos)} samples\n")
                
                logger.info(f"Statistiques du dataset sauvegardées dans {stats_file}")
            
            logger.info(f"Prétraitement WLASL terminé. {sum(len(videos) for videos in result.values())} vidéos traitées dans {len(result)} classes.")
            
            # Vérifier si des vidéos sont manquantes
            if len(preprocessor.missing_videos) > 0:
                logger.warning(f"{len(preprocessor.missing_videos)} vidéos manquantes détectées.")
                logger.info(f"Liste des vidéos manquantes sauvegardée dans {missing_log_path}")
                logger.info("Pour obtenir ces vidéos manquantes, consultez: https://github.com/dxli94/WLASL#requesting-missing--pre-processed-videos")
        else:
            logger.error(f"Fichier d'annotations WLASL introuvable: {wlasl_json_path}")
            return
    else:
        # Traiter les vidéos sans annotations WLASL
        result = preprocessor.process_dataset(raw_dir, processed_dir)
        logger.info(f"Prétraitement terminé. {len(result)} vidéos traitées.")


def find_missing_wlasl_videos(cfg: DictConfig):
    """
    Fonction pour identifier les vidéos manquantes du dataset WLASL.
    
    Args:
        cfg (DictConfig): Configuration hydra.
    """
    logger = get_logger(__name__)
    logger.info("Recherche des vidéos manquantes dans le dataset WLASL...")
    
    wlasl_json_path = cfg.get("data", {}).get("wlasl_json_path", None)
    raw_dir = cfg.data.raw_dir
    missing_log_path = cfg.get("data", {}).get("missing_videos_log", "data/missing_videos.txt")
    
    if not wlasl_json_path or not os.path.exists(wlasl_json_path):
        logger.error(f"Fichier d'annotations WLASL introuvable: {wlasl_json_path}")
        return
    
    if not os.path.exists(raw_dir):
        logger.error(f"Répertoire de données brutes introuvable: {raw_dir}")
        return
    
    # Initialiser le préprocesseur juste pour utiliser ses méthodes
    preprocessor = VideoPreprocessor(cfg, logger)
    
    # Charger les annotations
    wlasl_data = preprocessor.load_wlasl_json(wlasl_json_path)
    
    missing_videos = []
    total_videos = 0
    
    for entry in wlasl_data:
        gloss = entry["gloss"]
        for instance in entry["instances"]:
            total_videos += 1
            video_id = instance["video_id"]
            instance_id = instance["instance_id"]
            
            # Vérifier si la vidéo existe
            video_path = preprocessor._find_video_file(raw_dir, video_id)
            if not video_path:
                missing_videos.append({
                    "video_id": video_id,
                    "gloss": gloss,
                    "instance_id": instance_id
                })
    
    # Sauvegarder la liste des vidéos manquantes
    if missing_videos:
        with open(missing_log_path, 'w') as f:
            f.write("video_id,gloss,instance_id\n")
            for video in missing_videos:
                f.write(f"{video['video_id']},{video['gloss']},{video['instance_id']}\n")
        
        logger.info(f"Liste des {len(missing_videos)} vidéos manquantes sauvegardée dans {missing_log_path}")
        logger.info(f"Vidéos manquantes: {len(missing_videos)}/{total_videos} ({len(missing_videos)/total_videos*100:.2f}%)")
        logger.info("Pour obtenir ces vidéos manquantes, consultez: https://github.com/dxli94/WLASL#requesting-missing--pre-processed-videos")
    else:
        logger.info("Toutes les vidéos sont présentes!")


if __name__ == "__main__":
    # Pour des tests autonomes, vous pouvez exécuter ce script directement
    import hydra
    
    @hydra.main(config_path="../configs", config_name="config")
    def run_preprocessing(cfg: DictConfig):
        # Vérifier si l'utilisateur souhaite seulement trouver les vidéos manquantes
        if cfg.get("mode", {}).get("find_missing", False):
            find_missing_wlasl_videos(cfg)
        else:
            main(cfg)
    
    run_preprocessing() 