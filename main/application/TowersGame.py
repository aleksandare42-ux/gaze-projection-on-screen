import tkinter as tk
import time
import asyncio
import global_config as cfg
from pkgs import functions
try:
    from PIL import Image, ImageTk
except ImportError:
    print("Ошибка: Для работы с изображениями блоков требуется библиотека Pillow.")
    print("Установите ее: pip install Pillow")
    exit()


# ==================================
# Глобальные переменные (Placeholder)
# ==================================
BACKGROUND_IMG = "app_imgs/Gemini_Generated_Image_Troeshchina.png"  # Изображение фона
BLOCK_IMG = "app_imgs/Gemini_Generated_Image_metro_tram.png"
BLOCK_COLOR = "purple"  # Цвет блока по умолчанию
INITIAL_BLOCK_WIDTH = 400  # Начальная ширина блока
INITIAL_BLOCK_HEIGHT = 70 # Высота каждого блока
GAME_SPEED = 10  # Скорость движения блока
MOVE_DIRECTION = 1  # Начальное направление движения (1: вправо, -1: влево)

# ==================================
# Параметры игры
# ==================================
SCORE_MULTIPLIER = 10  # Множитель для подсчёта очков
MIN_BLOCK_WIDTH = 5  # Минимальная ширина блока для продолжения игры


class TowersGame(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Towers Game")
        self.attributes('-fullscreen', True)
        self.grab_set()
        self.focus_force()
        self.attributes('-topmost', True)
        cfg.pass_launcher_menu=True

        self.width = self.winfo_screenwidth()
        self.height = self.winfo_screenheight()

        # Canvas
        self.canvas = tk.Canvas(self, width=self.width, height=self.height, bg="#333333", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # --- Состояние игры ---
        self.current_block_width = INITIAL_BLOCK_WIDTH
        self.last_block_x_center = self.width // 2
        # blocks: список (ID, x_center, width)
        self.blocks = []
        self.current_moving_block = None
        self.current_block_x_center = 0
        self.direction = MOVE_DIRECTION
        # self.failures = 0  <-- УДАЛЕНО!
        self.score = 0
        self.running = False
        self.game_after_id = None

        #--- Атрибуты для изображений ---
        self.original_pil_image = None  # Исходное PIL изображение (для масштабирования)
        self.current_block_tk = None  # PhotoImage для *движущегося* блока
        self.settled_blocks_tk = []  # Список PhotoImage для *установленных* блоков (ВАЖНО!)
        self.bg_tk_img = None  # ФОНОВОЕ ИЗОБРАЖЕНИЕ

        # --- Внешнее управление ---
        self.external_drop_signal = False

        # --- Графические элементы ---
        self.score_text_id = None

        # Bindings
        self.bind("<space>", self.on_drop)
        self.bind("<Escape>", lambda e: self.close_and_restore())

        # +++ Загружаем изображение +++
        self.load_original_image()
        self.load_background()

        self.show_start_screen()

        self.show_start_screen()

    def load_original_image(self):
        """Загружает и масштабирует исходное изображение блока."""
        try:
            img = Image.open(BLOCK_IMG)
            # Масштабируем до начальных размеров игры
            self.original_pil_image = img.resize(
                (INITIAL_BLOCK_WIDTH, INITIAL_BLOCK_HEIGHT),
                Image.Resampling.LANCZOS  # Используем качественное сжатие
            )
        except FileNotFoundError:
            print(f"Ошибка: Изображение не найдено по пути: {BLOCK_IMG}")
            self.close_and_restore()
        except Exception as e:
            print(f"Ошибка при загрузке изображения: {e}")
            self.close_and_restore()

    def load_background(self):
        """Загружает, масштабирует и устанавливает фоновое изображение."""
        try:
            # Загружаем
            bg_pil_img = Image.open(BACKGROUND_IMG)
            # Масштабируем до размера экрана
            bg_pil_img = bg_pil_img.resize(
                (self.width, self.height),
                Image.Resampling.LANCZOS
            )
            self.bg_tk_img = ImageTk.PhotoImage(bg_pil_img)

            # Рисуем на canvas (в центре)
            self.canvas.create_image(
                self.width // 2, self.height // 2,
                image=self.bg_tk_img,
                tags="background"  # Тег, чтобы фон был "ниже" всего
            )
            # Отправляем фон на задний план
            self.canvas.tag_lower("background")

        except FileNotFoundError:
            print(f"Ошибка: Фоновое изображение не найдено: {BACKGROUND_IMG}")
            # Игра продолжится с серым фоном
        except Exception as e:
            print(f"Ошибка при загрузке фона: {e}")

    # -----------------------
    # Screens and Cleanup
    # -----------------------
    def clear_canvas(self):
        self.canvas.delete("all")
        if self.bg_tk_img:
            self.canvas.create_image(
                self.width // 2, self.height // 2,
                image=self.bg_tk_img,
                tags="background"
            )
            self.canvas.tag_lower("background")
        self.blocks = []
        self.current_moving_block = None
        self.score_text_id = None
        self.settled_blocks_tk = []
        self.current_block_tk = None

    def close_and_restore(self):
        self.destroy()
        self.master.deiconify()                     # Показать снова
        self.master.lift()                          # Поднять на верх
        self.master.attributes('-topmost', True)    # Временно сделать поверх всех
        self.cond_video_process = False
        cfg.initial_params_to_class()
        cfg.pass_launcher_menu = False
        cfg.main_process_loger.info("Towers game closed, returning to launcher. \n" \
        f"Params: EYE_ANALYSIS={cfg.CameraProcessor.EYE_ANALYSIS}, \nHAND_DETECTION={cfg.CameraProcessor.HAND_DETECTION}, " \
        f"\nOK_SIGHN_DETECTION={cfg.CameraProcessor.OK_SIGHN_DETECTION}, \nVICTORY_TOGETHER_SIGHN_DETECTION={cfg.CameraProcessor.VICTORY_TOGETHER_SIGHN_DETECTION}"\
            f"pass_launcher_menu={cfg.pass_launcher_menu}")
        # self.master.after(100, lambda: self.attributes('-topmost', False))
        self.destroy_self()
        
    def destroy_self(self):
        print("Deactivating...")
        for k in list(self.__dict__.keys()):
            delattr(self, k)

    def show_start_screen(self):
        self.stop_game_loop()
        self.clear_canvas()

        self.canvas.create_text(self.width // 2, self.height // 3, text="Towers (Stacker)", fill="white",
                                font=("Arial", 48, "bold"))
        rules = (
            "Правила:\n\n"
            "1. Сбрасывайте новый блок точно на предыдущий (клавиша ПРОБЕЛ/действие).\n"
            "2. Неточная посадка отрезает свисающую часть, уменьшая ширину следующего блока.\n"
            "3. Игра заканчивается, если вы промахнулись полностью или ширина блока стала < 5px."
        )
        self.canvas.create_text(self.width // 2, int(self.height * 0.55), text=rules, fill="white",
                                font=("Arial", 18), width=int(self.width * 0.7))
        ok_btn = tk.Button(self, text="Начать", font=("Arial", 20), command=self.start_game)
        self.canvas.create_window(self.width // 2, int(self.height * 0.75), window=ok_btn)

    def show_score_screen(self):
        self.stop_game_loop()
        self.clear_canvas()
        final_score = self.score * SCORE_MULTIPLIER
        text = f"Игра окончена. Высота башни: {self.score}\nФинальный счёт: {final_score}"
        self.canvas.create_text(self.width // 2, self.height // 3, text=text, fill="white",
                                font=("Arial", 40))
        restart_btn = tk.Button(self, text="Restart", font=("Arial", 18), command=self.start_game)
        exit_btn = tk.Button(self, text="Exit", font=("Arial", 18), command=self.close_and_restore)
        self.canvas.create_window(self.width // 2 - 100, int(self.height * 0.6), window=restart_btn)
        self.canvas.create_window(self.width // 2 + 100, int(self.height * 0.6), window=exit_btn)

    # -----------------------
    # Game lifecycle
    # -----------------------
    def start_game(self):
        self.clear_canvas()
        self.score = 0
        # self.failures = 0 <-- УДАЛЕНО!
        self.current_block_width = INITIAL_BLOCK_WIDTH
        self.last_block_x_center = self.width // 2

        self.draw_initial_base()

        self.running = True
        self.spawn_new_block()
        self.schedule_next_frame()

    def stop_game_loop(self):
        self.running = False
        if self.game_after_id is not None:
            try:
                self.after_cancel(self.game_after_id)
            except Exception:
                pass
            self.game_after_id = None

    # -----------------------
    # Drawing and Physics
    # -----------------------
    def draw_initial_base(self):
        """Рисует первый (базовый) блок."""
        x_center = self.width // 2
        y_bottom = self.height - 50
        # +++ Y-центр для create_image +++
        y_center = y_bottom - INITIAL_BLOCK_HEIGHT // 2

        # - # Рисуем базовый блок
        # - self.canvas.create_rectangle(
        # -     x_center - INITIAL_BLOCK_WIDTH // 2, y_bottom - INITIAL_BLOCK_HEIGHT,
        # -     x_center + INITIAL_BLOCK_WIDTH // 2, y_bottom,
        # -     fill=BLOCK_COLOR, tags="block"
        # - )

        # +++ Рисуем базовое ИЗОБРАЖЕНИЕ +++
        # original_pil_image уже имеет нужный размер
        base_tk_img = ImageTk.PhotoImage(self.original_pil_image)
        self.canvas.create_image(
            x_center, y_center,
            image=base_tk_img,
            tags="block"
        )
        # +++ СОХРАНЯЕМ ССЫЛКУ +++
        self.settled_blocks_tk.append(base_tk_img)

        # Добавляем его в список как предыдущий
        self.blocks.append((
            None,  # ID не так важен, если мы не двигаем старые блоки
            x_center,
            INITIAL_BLOCK_WIDTH
        ))
        self.last_block_x_center = x_center

    def spawn_new_block(self):
        """Создает новый блок для движения."""
        self.current_block_x_center = self.current_block_width // 2
        self.direction = MOVE_DIRECTION

        # - self.current_moving_block = self.canvas.create_rectangle(
        # -     0, 0, 0, 0,
        # -     fill=BLOCK_COLOR, tags="moving_block"
        # - )

        # +++ Создаем масштабированное изображение для движущегося блока +++
        # Убедимся, что ширина не стала 0 или отрицательной
        width = max(1, int(self.current_block_width))
        height = INITIAL_BLOCK_HEIGHT

        scaled_pil_img = self.original_pil_image.resize(
            (width, height),
            Image.Resampling.LANCZOS
        )
        self.current_block_tk = ImageTk.PhotoImage(scaled_pil_img)

        self.current_moving_block = self.canvas.create_image(
            0, 0,  # Координаты будут заданы в update_graphics
            image=self.current_block_tk,
            tags="moving_block"
        )

        self.update_graphics()
        self.update_score_display()

    def get_block_y_pos(self, index):
        """Вычисляет Y-координату нижнего края блока по индексу."""
        base_y = self.height - 50
        return base_y - (len(self.blocks) * INITIAL_BLOCK_HEIGHT)

    def update_graphics(self):
        """Обновление положения движущегося блока."""

        y_bottom = self.get_block_y_pos(len(self.blocks))
        # +++ Нужен Y-центр +++
        y_center = y_bottom - INITIAL_BLOCK_HEIGHT // 2

        # - x_left = self.current_block_x_center - self.current_block_width // 2
        # - x_right = self.current_block_x_center + self.current_block_width // 2
        # -
        # - self.canvas.coords(
        # -     self.current_moving_block,
        # -     x_left, y_top, x_right, y_bottom
        # - )

        # +++ Обновляем (x, y) центра +++
        self.canvas.coords(
            self.current_moving_block,
            self.current_block_x_center, y_center
        )

    def update_score_display(self):
        """Обновляет отображение счёта."""
        if self.score_text_id:
            self.canvas.delete(self.score_text_id)

        final_score = self.score * SCORE_MULTIPLIER
        # Убрана информация об ошибках
        text = f"Счёт: {final_score}"
        self.score_text_id = self.canvas.create_text(
            self.width // 2, 50, text=text, fill="white",
            font=("Arial", 24), justify=tk.CENTER
        )

    def schedule_next_frame(self):
        self.game_after_id = self.after(16, self.game_loop)

    def game_loop(self):
        if not self.running:
            return

        # 1. Движение блока
        self.current_block_x_center += self.direction * GAME_SPEED

        # Проверка границ (отскок от стен)
        half_width = self.current_block_width // 2
        if self.current_block_x_center + half_width >= self.width:
            self.direction = -1
        elif self.current_block_x_center - half_width <= 0:
            self.direction = 1

        # 2. Обновление графики
        self.update_graphics()

        # 3. Продолжение цикла
        self.schedule_next_frame()

    # -----------------------
    # Drop/Collision Logic
    # -----------------------
    def on_drop(self, event=None):
        """Обработчик сброса блока (клавиша или внешнее управление)."""
        if not self.running or self.current_moving_block is None:
            return

        # 1. Отменяем следующий запланированный кадр
        if self.game_after_id is not None:
            self.after_cancel(self.game_after_id)
            self.game_after_id = None

        # 2. Вычисляем посадку.
        # Если игра закончится, process_drop вызовет show_score_screen,
        # который сам установит self.running = False.
        self.process_drop()

        # 3. Если игра не была остановлена (т.е. нет Game Over)
        if self.running:
            # Запускаем новый блок и возобновляем цикл
            self.spawn_new_block()
            self.schedule_next_frame()  # Возобновляем game_loop

    def process_drop(self):
        """Логика обработки посадки блока. (Остается без изменений)"""

        last_block_data = self.blocks[-1]
        last_x_center = last_block_data[1]
        last_width = last_block_data[2]

        # Границы предыдущего блока
        last_left = last_x_center - last_width // 2
        last_right = last_x_center + last_width // 2

        # Границы текущего (движущегося) блока
        current_left = self.current_block_x_center - self.current_block_width // 2
        current_right = self.current_block_x_center + self.current_block_width // 2

        # 1. Считаем перекрытие
        overlap_left = max(last_left, current_left)
        overlap_right = min(last_right, current_right)

        overlap_width = overlap_right - overlap_left

        # 2. Проверка на полный промах (Game Over)
        if overlap_width <= 0:
            # Полный промах -> Game Over
            self.show_score_screen()
            return

        # 3. Успешная посадка / Обрезка

        new_width = overlap_width
        new_x_center = (overlap_left + overlap_right) // 2

        self.canvas.delete(self.current_moving_block)
        self.current_block_tk = None  # Очищаем ссылку на движущийся блок

        # - # Рисуем новый (обрезанный) блок на месте посадки
        y_bottom = self.get_block_y_pos(len(self.blocks))
        # - y_top = y_bottom - INITIAL_BLOCK_HEIGHT
        # -
        # - settled_block_id = self.canvas.create_rectangle(
        # -     new_x_center - new_width // 2, y_top,
        # -     new_x_center + new_width // 2, y_bottom,
        # -     fill=BLOCK_COLOR, tags="block"
        # - )

        # +++ Рисуем новое (масштабированное) ИЗОБРАЖЕНИЕ +++
        y_center = y_bottom - INITIAL_BLOCK_HEIGHT // 2

        # Убедимся, что ширина > 0
        width = max(1, int(new_width))
        height = INITIAL_BLOCK_HEIGHT

        settled_pil_img = self.original_pil_image.resize(
            (width, height),
            Image.Resampling.LANCZOS
        )
        settled_tk_img = ImageTk.PhotoImage(settled_pil_img)

        settled_block_id = self.canvas.create_image(
            new_x_center, y_center,
            image=settled_tk_img,
            tags="block"
        )
        # +++ СОХРАНЯЕМ ССЫЛКУ +++
        self.settled_blocks_tk.append(settled_tk_img)

        # Обновление состояния для следующего блока
        self.current_block_width = new_width
        self.last_block_x_center = new_x_center
        self.blocks.append((settled_block_id, new_x_center, new_width))
        self.score += 1

        # 4. Проверка на Game Over: блок стал слишком тонким
        if new_width < MIN_BLOCK_WIDTH:
            self.show_score_screen()
            return

        self.update_score_display()

    # -----------------------
    # External API
    # -----------------------
    def set_control_action(self, action: str):
        """
        Метод для внешнего управления.

        Параметры:
            action: 'drop' для сброса блока.
        """
        if action == 'drop' and self.running:
            # on_drop() должен быть вызван в главном потоке Tkinter
            self.after_idle(self.on_drop)

    def run_camera_analysis(self):
        # self.cap = cv2.VideoCapture(0)
        # # self.camera_analysis_update()
        # if cfg.HAND_DETECTION:
        #     self.HandRecognizer = hand_recognontion.hand_processing()
        # if cfg.EYE_ANALYSIS:
        #     self.EyeAnalyzer = eye_analys.SightDetectionAsync()
        functions.run_async(self, self.camera_analysis_async())
    
    async def camera_analysis_async(self):
        # time_start = time.time()
        print(f"Starting {self.__name__} camera analysis...")
        print(self.width)
        cfg.main_process_loger.info("Starting Flappy-bird camera analysis...")
        frame_count = 0
        cond= True
        cfg.CameraProcessor.EYE_ANALYSIS=False
        # launching = False
        while self.cond_video_process:
            # if cfg.VICTORY_TOGETHER_SIGHN_DETECTION:
            #     x, y, = cfg.CameraProcessor.x, cfg.CameraProcessor.y
            #     if x is not None and y is not None and not cfg.CameraProcessor.last_command_executed:
            #         # x*=self.width
            #         print("Camera x,y:", x, y, cfg.CameraProcessor.last_command_executed)
            #         if self.x_rocket_position is not None:
            #             delta_x = x - self.x_rocket_position
            #             delta_x*=1.5
            #             self.x_rocket_position+=delta_x
            #             self.paddle_x+=delta_x
            #             if self.paddle_x < self.paddle_width//2:
            #                 self.paddle_x = self.paddle_width//2
            #             if self.paddle_x > self.width - self.paddle_width//2:
            #                 self.paddle_x = self.width - self.paddle_width//2
            #             print("Trying to move rocket to:", self.x_rocket_position)
            #             self.draw_paddle()
            #         else:
            #             self.x_rocket_position = self.paddle_x
            #         cfg.CameraProcessor.last_command_executed = True
            #     else: self.x_rocket_position=None
            # else: self.x_rocket_position=None
            frame_count += 1
            await asyncio.sleep(0.05)


# -----------------------
# Запуск
# -----------------------
if __name__ == "__main__":
    root = tk.Tk()
    game = TowersGame(root)
    root.mainloop()