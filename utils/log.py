import logging
import os


def log(output):
    # 配置控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    # 配置文件输出
    file_handler = logging.FileHandler(os.path.join(output, 'log.txt'), mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # 创建一个日志器实例
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 添加控制台和文件处理器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
