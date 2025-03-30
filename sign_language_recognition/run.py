"""
Entrée principale du système de reconnaissance de langue des signes.
"""
import os
import sys
import subprocess
import dotenv

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from src.utils.logger import get_logger

# Charger les variables d'environnement si présentes
dir_path = os.path.dirname(os.path.realpath(__file__))
env_path = os.path.join(dir_path, "environment.env")
if os.path.exists(env_path):
    dotenv.load_dotenv(env_path, override=True)

@hydra.main(config_path="configs", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    """
    Point d'entrée principal:
        1) Si test_mode = True, exécuter les tests et terminer
        2) Sinon, charger dynamiquement le module et la fonction depuis model_dispatch
    
    Args:
        cfg (DictConfig): Configuration chargée par Hydra.
    """
    logger = get_logger(__name__)
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    # 1) Si test_mode est activé
    if cfg.mode.test_mode:
        logger.info("Mode test activé.")
        test_dir = cfg.get("mode", {}).get("test_dir", "tests")
        test_file = cfg.get("mode", {}).get("test_file", "")

        if test_file:
            # Exécuter un fichier de test spécifique
            test_path = os.path.join(test_dir, test_file)
            logger.info(f"Exécution des tests pour le fichier: {test_path}")
            cmd = f"python -m pytest {test_path} -v"
        else:
            # Exécuter tous les tests dans le répertoire
            logger.info(f"Exécution de tous les tests dans {test_dir}")
            cmd = f"python -m pytest {test_dir} -v"

        # Exécuter les tests en sous-processus
        ret = subprocess.call(cmd, shell=True)
        logger.info(f"Tests terminés avec le code de sortie {ret}")
        return

    # 2) Mode normal: charger dynamiquement depuis model_dispatch
    model_name = cfg.mode.selected_model
    if not model_name:
        logger.error("mode.selected_model doit être spécifié dans la configuration.")
        sys.exit(1)

    # Initialiser wandb si activé
    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.name,
            tags=cfg.wandb.tags,
            notes=cfg.wandb.notes,
            config=OmegaConf.to_container(cfg, resolve=True)
        )
        logger.info(f"WandB initialisé: {wandb.run.name}")

    dispatch_dict = cfg.model_dispatch
    if not dispatch_dict:
        logger.error("model_dispatch n'est pas défini dans la configuration.")
        sys.exit(1)

    import importlib

    if model_name in dispatch_dict:
        path_func_str = dispatch_dict[model_name]  # e.g. "src.data_preprocessing:main"
        module_path, func_name = path_func_str.split(":")

        try:
            mod = importlib.import_module(module_path)
            main_func = getattr(mod, func_name)
            logger.info(f"Dispatching vers {path_func_str} avec model_name={model_name}")
            main_func(cfg)
        except ImportError as e:
            logger.error(f"Impossible d'importer {module_path}: {e}")
            sys.exit(1)
        except AttributeError as e:
            logger.error(f"Fonction {func_name} introuvable dans {module_path}: {e}")
            sys.exit(1)
    else:
        logger.error(f"Aucune entrée correspondante dans model_dispatch pour mode.selected_model={model_name}")
        sys.exit(1)

if __name__ == "__main__":
    main()
    if wandb.run is not None:
        wandb.finish()
