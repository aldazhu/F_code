import logging
import os

# 删除log
log_file = "logs/log.txt"
if os.path.exists(log_file):
    os.remove(log_file)

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
# set the file file_handler
file_handler = logging.FileHandler("logs/log.txt")
file_handler.setLevel(logging.INFO)
# set the console_handler file_handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
# set the formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

all = [logger]