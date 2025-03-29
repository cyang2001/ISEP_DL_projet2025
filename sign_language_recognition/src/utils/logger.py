"""
Module de journalisation pour le projet de reconnaissance de langue des signes.
"""
import logging
import logging.handlers
import os
import wandb
import threading
from datetime import datetime
from multiprocessing import current_process


def get_logger(name=__name__, log_queue=None):
    """
    Crée et configure un logger pour l'application.
    
    Cette fonction configure un logger avec des gestionnaires appropriés pour la console,
    les fichiers et Weights & Biases. Elle prend également en charge les environnements
    multiprocessus.
    
    Args:
        name (str): Nom du logger, généralement le nom du module.
        log_queue (Queue, optional): File d'attente pour la journalisation multiprocessus.
        
    Returns:
        logging.Logger: Un logger configuré.
    """
    logger = logging.getLogger(name)
    logger.propagate = False

    # Définir le niveau de journalisation
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    numeric_level = getattr(logging, log_level, logging.INFO)
    logger.setLevel(numeric_level)

    if not logger.handlers:
        # Déterminer si nous sommes dans le processus principal ou un processus enfant
        is_main_process = (current_process().name == 'MainProcess')
        # Vérifier si log_queue est fourni (indique le multitraitement)
        is_multiprocessing = (log_queue is not None)

        # Configurer le répertoire et le gestionnaire de fichiers
        log_dir = os.getenv('LOG_DIR', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'{name}_{datetime.now():%Y-%m-%d}.log')

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(processName)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        # Configurer le gestionnaire de console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(processName)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)

        # Gestionnaire Wandb
        class WandbHandler(logging.Handler):
            """
            Gestionnaire de journalisation personnalisé pour Weights & Biases.
            """
            def emit(self, record):
                """
                Émet un enregistrement de journal vers Weights & Biases.
                
                Args:
                    record (LogRecord): L'enregistrement de journal à émettre.
                """
                try:
                    if wandb.run is None:
                        return
                    log_entry = {
                        'log_level': record.levelname,
                        'logger_name': record.name,
                        'message': record.getMessage(),
                        'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                    }

                    if record.exc_info:
                        log_entry['exception'] = logging.Formatter().formatException(record.exc_info)

                    log_key = f"{record.levelname.lower()}_log"

                    wandb.log({
                        log_key: log_entry,
                        f"{log_key}_message": record.getMessage(),
                        'log_step': wandb.run.step if wandb.run else 0
                    })
                except Exception as e:
                    print(f"Failed to log to wandb: {e}")

        wandb_handler = WandbHandler()
        wandb_handler.setLevel(logging.DEBUG)
        wandb_formatter = logging.Formatter('%(asctime)s - %(processName)s - %(name)s - %(levelname)s - %(message)s')
        wandb_handler.setFormatter(wandb_formatter)

        if is_multiprocessing and not is_main_process:
            # Dans un processus enfant pendant le multitraitement
            # Utiliser QueueHandler pour envoyer des journaux au processus principal
            queue_handler = logging.handlers.QueueHandler(log_queue)
            queue_handler.setLevel(numeric_level)
            logger.addHandler(queue_handler)
        else:
            # Dans le processus principal ou en mode mono-thread
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            logger.addHandler(wandb_handler)

    return logger


def start_log_listener(log_queue):
    """
    Démarre un écouteur de journalisation dans un thread séparé.
    
    Cette fonction configure un logger dédié qui traite les enregistrements 
    de journal provenant d'une file d'attente, utile pour la journalisation 
    multiprocessus.
    
    Args:
        log_queue (Queue): File d'attente contenant les enregistrements de journal.
        
    Returns:
        threading.Thread: Thread de l'écouteur de journalisation.
    """
    # Créer un logger séparé, plutôt que d'utiliser le logger racine
    listener_logger = logging.getLogger('log_listener')
    listener_logger.propagate = False  # Empêcher les journaux de se propager au logger racine
    listener_logger.setLevel(logging.DEBUG)

    # Vérifier si des gestionnaires ont déjà été ajoutés, éviter les doublons
    if not listener_logger.handlers:
        # Configurer les gestionnaires
        log_dir = os.getenv('LOG_DIR', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'log_{datetime.now():%Y-%m-%d}.log')

        # Gestionnaire de fichiers
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(processName)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        # Gestionnaire de console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(processName)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)

        # Gestionnaire Wandb
        class WandbHandler(logging.Handler):
            """
            Gestionnaire de journalisation personnalisé pour Weights & Biases.
            """
            def emit(self, record):
                """
                Émet un enregistrement de journal vers Weights & Biases.
                
                Args:
                    record (LogRecord): L'enregistrement de journal à émettre.
                """
                try:
                    if wandb.run is None:
                        return
                    log_entry = {
                        'log_level': record.levelname,
                        'logger_name': record.name,
                        'message': record.getMessage(),
                        'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                    }

                    if record.exc_info:
                        log_entry['exception'] = logging.Formatter().formatException(record.exc_info)

                    log_key = f"{record.levelname.lower()}_log"

                    wandb.log({
                        log_key: log_entry,
                        f"{log_key}_message": record.getMessage(),
                        'log_step': wandb.run.step if wandb.run else 0
                    })
                except Exception as e:
                    print(f"Failed to log to wandb: {e}")

        wandb_handler = WandbHandler()
        wandb_handler.setLevel(logging.DEBUG)
        wandb_formatter = logging.Formatter('%(asctime)s - %(processName)s - %(name)s - %(levelname)s - %(message)s')
        wandb_handler.setFormatter(wandb_formatter)

        # Ajouter des gestionnaires au listener_logger
        listener_logger.addHandler(file_handler)
        listener_logger.addHandler(console_handler)
        listener_logger.addHandler(wandb_handler)

    def listener():
        """Fonction d'écoute qui traite les enregistrements de journal de la file d'attente."""
        while True:
            try:
                record = log_queue.get()
                if record is None:  # Signal d'arrêt
                    break
                listener_logger.handle(record)
            except Exception:
                import traceback
                traceback.print_exc()

    listener_thread = threading.Thread(target=listener)
    listener_thread.daemon = True
    listener_thread.start()
    return listener_thread 