import shutil
import logging
from pathlib import Path
from datetime import datetime
import argparse

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("backup.log")
    ]
)
logger = logging.getLogger("BackupTool")

DATA_DIR = Path("data")
BACKUP_DIR = Path("backups")

def create_backup():
    """
    Creates a timestamped zip archive of the data/ directory.
    Uses shutil.make_archive for a crash-consistent snapshot.
    """
    if not DATA_DIR.exists():
        logger.error(f"Data directory {DATA_DIR} does not exist. Nothing to backup.")
        return False
    
    BACKUP_DIR.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"knowledge_backup_{timestamp}"
    backup_path = BACKUP_DIR / backup_name
    
    try:
        logger.info(f"Starting backup of {DATA_DIR}...")
        
        # make_archive works on the content of the directory
        # root_dir=DATA_DIR means we zip content *inside* data, not data folder itself usually
        # But for restoration ease, usually we want to zip the folder structure.
        # Let's zip the folder itself.
        
        archive_name = shutil.make_archive(str(backup_path), 'zip', root_dir=".", base_dir="data")
        
        logger.info(f"Backup created successfully: {archive_name}")
        # Clean up old backups? (Optional enhancement)
        return True
    except Exception as e:
        logger.error(f"Backup failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backup Knowledge Data")
    parser.parse_args()
    
    create_backup()
