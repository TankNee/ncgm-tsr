from loguru import logger

logger.add("./logs/ncgm.log", rotation="1 day", compression="zip")