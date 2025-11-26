import logging
import camera_processing


EYE_ANALYSIS = True
HAND_DETECTION = True
OK_SIGHN_DETECTION = True
VICTORY_TOGETHER_SIGHN_DETECTION = True
CLIP_ANALYZER:bool = True


pass_launcher_menu = False
degue_size:int= 8

camera_logger = logging.getLogger("Camera_logger")
coordinate_logger = logging.getLogger("Coordinate_logger")

CameraProcessor = camera_processing.CameraProcessor(
    logger=camera_logger,
    HAND_DETECTION=HAND_DETECTION,
    EYE_ANALYSIS=EYE_ANALYSIS,
    OK_SIGHN_DETECTION=OK_SIGHN_DETECTION,
    VICTORY_TOGETHER_SIGHN_DETECTION=VICTORY_TOGETHER_SIGHN_DETECTION
)
CameraProcessor.run_camera_analysis()


def initial_params():
    global EYE_ANALYSIS
    global HAND_DETECTION
    global OK_SIGHN_DETECTION
    global VICTORY_TOGETHER_SIGHN_DETECTION
    EYE_ANALYSIS = False
    HAND_DETECTION = True
    OK_SIGHN_DETECTION = True
    VICTORY_TOGETHER_SIGHN_DETECTION = True

def initial_params_to_class():
    global CameraProcessor
    global EYE_ANALYSIS
    global HAND_DETECTION
    global OK_SIGHN_DETECTION
    global VICTORY_TOGETHER_SIGHN_DETECTION
    CameraProcessor.EYE_ANALYSIS = False
    CameraProcessor.HAND_DETECTION = True
    CameraProcessor.OK_SIGHN_DETECTION = True
    CameraProcessor.VICTORY_TOGETHER_SIGHN_DETECTION = True

def print_current_params():
    global EYE_ANALYSIS
    global HAND_DETECTION
    global OK_SIGHN_DETECTION
    global VICTORY_TOGETHER_SIGHN_DETECTION
    print("Current parameters:")
    print(f"EYE_ANALYSIS: {EYE_ANALYSIS}")
    print(f"HAND_DETECTION: {HAND_DETECTION}")
    print(f"OK_SIGHN_DETECTION: {OK_SIGHN_DETECTION}")
    print(f"VICTORY_TOGETHER_SIGHN_DETECTION: {VICTORY_TOGETHER_SIGHN_DETECTION}")


def setup_main_logger(
    name="main",
    log_file="main_logger.txt",
    file_level=logging.DEBUG,
    console_level=logging.INFO
):
    # создаём логер
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # общий уровень

    # === формат вывода ===
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # === обработчик для файла ===
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)

    # === обработчик для консоли ===
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)

    # === добавляем обработчики ===
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

main_process_loger = setup_main_logger(
    name="Main_process_logger")
main_process_loger.info("Main process logger setuped.")
print_current_params()