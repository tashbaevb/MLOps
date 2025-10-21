import os, time, logging, threading
from utils import config
from ml import retrain

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

last_mtime = None


def watch_dataset():
    global last_mtime
    if not os.path.exists(config.DATA_FILE):
        logger.warning("Dataset not found.")
        return
    last_mtime = os.path.getmtime(config.DATA_FILE)
    logger.info(f"👀 Watching for changes in {config.DATA_FILE}")

    while True:
        time.sleep(10)
        try:
            mtime = os.path.getmtime(config.DATA_FILE)
            if mtime != last_mtime:
                logger.info("Dataset changed — triggering retrain.")
                retrain.main()
                print("API MODEL_DIR:", config.MODEL_DIR)
                last_mtime = mtime
        except Exception as e:
            logger.error(f"Watcher error: {e}")


def start_watcher():
    thread = threading.Thread(target=watch_dataset, daemon=False)
    thread.start()
    logger.info("Watcher started.")
    thread.join()


if __name__ == "__main__":
    start_watcher()
