import cv2
import os
import csv
import random, screeninfo
from datetime import datetime

# === Настройки ===
RECT_SIZE = 20  # размер белого прямоугольника (20x20)
DATASET_DIR = "dataset"

# === Создание директорий и csv ===
def create_dataset_dirs(base_dir=DATASET_DIR):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_path = os.path.join(os.getcwd(), base_dir)
    os.makedirs(dataset_path, exist_ok=True)

    images_dir = os.path.join(dataset_path, f"images_{timestamp}")
    labels_dir = os.path.join(dataset_path, f"labels_{timestamp}")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    csv_path = os.path.join(dataset_path, f"dataset_{timestamp}.csv")
    csv_file = open(csv_path, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["image_name", "x_center_norm", "y_center_norm"])

    return images_dir, labels_dir, csv_file, csv_writer

# === Основная программа ===
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Не удалось открыть камеру.")
        return

    images_dir, labels_dir, csv_file, csv_writer = create_dataset_dirs()
    screen = screeninfo.get_monitors()[0]
    screen_width, screen_height = screen.width, screen.height
    print("✅ Программа запущена.")
    print("Нажмите 'D' — сохранить изображение и аннотацию.")
    print("Нажмите 'L' — пропустить текущую позицию.")
    print("Нажмите 'ESC' — выйти из программы.")

    while True:
        # Создаем черный экран
        frame = 255 * np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

        # Случайная позиция прямоугольника
        rect_x = random.randint(0, screen_width - RECT_SIZE)
        rect_y = random.randint(0, screen_height - RECT_SIZE)

        # Рисуем прямоугольник
        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + RECT_SIZE, rect_y + RECT_SIZE), (255, 255, 255), -1)

        # Показываем экран
        cv2.namedWindow("Dataset Generator", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Dataset Generator", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Dataset Generator", frame)

        # Ждем клавишу
        key = cv2.waitKey(0) & 0xFF

        # Выход
        if key == 27:  # ESC
            break

        # Пропустить текущую позицию
        elif key in [ord('l'), ord('L')]:
            continue

        # Сохранить фото и аннотацию
        elif key in [ord('d'), ord('D')]:
            ret, img = cap.read()
            if not ret:
                print("⚠ Не удалось сделать снимок.")
                continue

            # Имя файла
            img_name = f"img_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
            img_path = os.path.join(images_dir, img_name)
            label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + ".txt")

            # Сохраняем изображение
            cv2.imwrite(img_path, img)

            # Нормализованные координаты центра
            x_center = (rect_x + RECT_SIZE / 2) / screen_width
            y_center = (rect_y + RECT_SIZE / 2) / screen_height

            # Сохраняем аннотацию
            with open(label_path, "w") as f:
                f.write(f"{x_center:.6f} {y_center:.6f}\n")

            # Пишем в CSV
            csv_writer.writerow([img_name, f"{x_center:.6f}", f"{y_center:.6f}"])

            print(f"💾 Сохранено: {img_name} — ({x_center:.4f}, {y_center:.4f})")

    # Завершение
    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()
    print("🚪 Выход. Датасет сохранен.")

# === Импорт numpy после функций, чтобы не мешать объявлениям ===
import numpy as np

if __name__ == "__main__":
    main()
