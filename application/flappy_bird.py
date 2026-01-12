import tkinter as tk
import time
from PIL import Image, ImageTk
import random
import asyncio
from pkgs import functions
import global_config as cfg

# ==================================
# Глобальные переменные (Placeholder)
# ==================================
BACKGROUND_IMG = "app_imgs/flappy_bg.png"  # Изображение фона
BIRD_IMG = "app_imgs/bird.png"  # Изображение птицы
PIPE_TOP_IMG = "app_imgs/pipe_top.png"  # Изображение верхней трубы
PIPE_BOTTOM_IMG = "app_imgs/pipe_bottom.png"  # Изображение нижней трубы

# ==================================
# Параметры игры (Можно настраивать)
# ==================================
BIRD_SIZE = 80
BIRD_RADIUS = BIRD_SIZE // 2
GRAVITY = 1.0
JUMP_STRENGTH = -15
PIPE_SPEED = 5
PIPE_GAP = 400
PIPE_WIDTH = 200
PIPE_WOBBLE = 40


class flappy_bird(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Flappy Bird")
        self.attributes('-fullscreen', True)
        # self.grab_set()
        self.focus_force()
        self.attributes('-topmost', True)
        cfg.pass_launcher_menu=True
        cfg.CameraProcessor.CLIP_ANALYZER=cfg.CLIP_ANALYZER
        self.cond_video_process = True

        self.width = self.winfo_screenwidth()
        self.height = self.winfo_screenheight()

        # Canvas
        self.canvas = tk.Canvas(self, width=self.width, height=self.height, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # --- Загрузка изображений ---
        self.bg_photo = None
        self.bird_photo = None
        # Оригинальные PIL.Image для динамического масштабирования труб
        self.pipe_top_original_img = None
        self.pipe_bottom_original_img = None
        self.load_images()

        # --- Состояние игры ---
        self.bird_y = self.height // 2
        self.bird_vy = 0
        self.score = 0
        self.running = False
        self.game_after_id = None
        # pipes: (id_top, id_bottom, x_center, photo_top, photo_bottom)
        self.pipes = []
        self.pipe_spawn_counter = 0

        # --- Внешнее управление ---
        self.external_jump_signal = False

        # --- Графические элементы ---
        self.bg_image_id = None
        self.bird_obj = None
        self.score_text_id = None

        # Bindings
        self.bind("<space>", self.on_jump)
        self.bind("<Escape>", lambda e: self.close_and_restore())

        self.run_camera_analysis()
        self.show_start_screen()

    # -----------------------
    # Setup
    # -----------------------
    def load_images(self):
        """Загружает все необходимые изображения с диска."""
        try:
            # Фон
            bg = Image.open(BACKGROUND_IMG)
            self.bg_photo = ImageTk.PhotoImage(bg.resize((self.width, self.height)))

            # Птица
            bird = Image.open(BIRD_IMG).convert("RGBA")
            self.bird_photo = ImageTk.PhotoImage(bird.resize((BIRD_SIZE, BIRD_SIZE)))

            # Трубы (Сохраняем как PIL.Image для динамического масштабирования)
            pipe_top = Image.open(PIPE_TOP_IMG).convert("RGBA")
            # Масштабируем по ширине, высоту сохраняем для будущих resize
            self.pipe_top_original_img = pipe_top.resize((PIPE_WIDTH, pipe_top.height))

            pipe_bottom = Image.open(PIPE_BOTTOM_IMG).convert("RGBA")
            self.pipe_bottom_original_img = pipe_bottom.resize((PIPE_WIDTH, pipe_bottom.height))

        except FileNotFoundError as e:
            print(f"Ошибка загрузки файла изображения: {e}. Используем заполнители (заглушки).")
        except Exception as e:
            print(f"Непредвиденная ошибка при загрузке изображений: {e}")

    # -----------------------
    # Screens and Cleanup
    # -----------------------
    def clear_canvas(self):
        self.canvas.delete("all")
        self.bg_image_id = None
        self.bird_obj = None
        self.score_text_id = None
        self.pipes = []

    def close_and_restore(self):
        # Сначала сбрасываем все флаги
        self.cond_video_process = False
        cfg.pass_launcher_menu = False
        cfg.CameraProcessor.CLIP_ANALYZER=False
        cfg.CameraProcessor.last_command_executed = True
        cfg.CameraProcessor.OK_sighn_detected = False
        
        # Потом закрываем окно
        self.destroy()
        self.master.deiconify()
        self.master.lift()
        self.master.attributes('-topmost', True)
        
        # Сбрасываем параметры конфига
        cfg.initial_params_to_class()
        
        cfg.main_process_loger.info("Flappy-bird game closed...")
        self.destroy_self()

    def destroy_self(self):
        print("Deactivating...")
        for k in list(self.__dict__.keys()):
            delattr(self, k)

    def show_start_screen(self):
        self.stop_game_loop()
        self.clear_canvas()

        if self.bg_photo:
            self.bg_image_id = self.canvas.create_image(0, 0, image=self.bg_photo, anchor="nw")
        else:
            self.canvas.configure(bg="skyblue")

        self.canvas.create_text(self.width // 2, self.height // 3, text="Flappy Bird", fill="white",
                                font=("Arial", 48, "bold"))
        ok_btn = tk.Button(self, text="Начать", font=("Arial", 20), command=self.start_game)
        self.canvas.create_window(self.width // 2, int(self.height * 0.6), window=ok_btn)

    def show_score_screen(self):
        self.stop_game_loop()
        self.clear_canvas()
        if self.bg_photo:
            self.bg_image_id = self.canvas.create_image(0, 0, image=self.bg_photo, anchor="nw")

        text = f"Игра окончена. Счёт: {self.score}"
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
        self.running = True

        if self.bg_photo:
            self.bg_image_id = self.canvas.create_image(0, 0, image=self.bg_photo, anchor="nw")

        self.bird_y = self.height // 2
        self.bird_vy = 0

        bird_x = 50
        if self.bird_photo:
            self.bird_obj = self.canvas.create_image(bird_x, self.bird_y, image=self.bird_photo)
        else:
            self.bird_obj = self.canvas.create_oval(
                bird_x - BIRD_RADIUS, self.bird_y - BIRD_RADIUS,
                bird_x + BIRD_RADIUS, self.bird_y + BIRD_RADIUS,
                fill="yellow"
            )

        self.score_text_id = self.canvas.create_text(self.width - 50, 50, text="Score: 0",
                                                     fill="white", font=("Arial", 30), anchor="ne")

        self.pipes = []
        self.pipe_spawn_counter = 0

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
    # Drawing and physics
    # -----------------------
    def schedule_next_frame(self):
        self.game_after_id = self.after(16, self.game_loop)

    def game_loop(self):
        if not self.running:
            return

        # 1. Обработка внешнего прыжка
        if self.external_jump_signal:
            self.apply_jump()
            self.external_jump_signal = False

        # 2. Физика птицы
        self.bird_vy += GRAVITY
        if self.bird_vy > 25:
            self.bird_vy = 25

        self.bird_y += self.bird_vy

        # Ограничиваем птицу экраном (без смерти)
        bird_x_center = 50
        if self.bird_y < BIRD_RADIUS:
            self.bird_y = BIRD_RADIUS
            if self.bird_vy < 0:
                self.bird_vy = 0
        elif self.bird_y > self.height - BIRD_RADIUS:
            self.bird_y = self.height - BIRD_RADIUS
            if self.bird_vy > 0:
                self.bird_vy = 0

        # 3. Спавн и движение труб
        self.update_pipes()

        # 4. Проверка коллизий
        if self.check_collisions():
            self.show_score_screen()
            return

        # 5. Обновление графики
        self.update_graphics()

        # 6. Продолжение цикла
        self.schedule_next_frame()

    def update_pipes(self):
        """Двигает трубы и создает новые."""
        pipes_to_keep = []
        # Теперь итерируемся по кортежу (id_top, id_bottom, x_center, photo_top, photo_bottom)
        for top_id, bottom_id, x_center, photo_top, photo_bottom in self.pipes:
            new_x = x_center - PIPE_SPEED

            if new_x < -PIPE_WIDTH:
                self.canvas.delete(top_id)
                self.canvas.delete(bottom_id)
                # Удалять photo_top/photo_bottom не обязательно, но это освободит память
                # (хотя Python это сделает сам, когда ссылка будет удалена из списка)
                self.score += 1
                self.canvas.itemconfig(self.score_text_id, text=f"Score: {self.score}")
            else:
                self.canvas.move(top_id, -PIPE_SPEED, 0)
                self.canvas.move(bottom_id, -PIPE_SPEED, 0)
                pipes_to_keep.append((top_id, bottom_id, new_x, photo_top, photo_bottom))

        self.pipes = pipes_to_keep

        self.pipe_spawn_counter += 1
        if self.pipe_spawn_counter >= 200:
            self.spawn_pipe()
            self.pipe_spawn_counter = 0

    def spawn_pipe(self):
        """Создает пару труб, используя изображения."""

        # 1. Определяем геометрию
        import random
        min_h = 50
        max_h = self.height - 50 - PIPE_GAP
        top_pipe_height = random.randint(min_h, max_h)
        bottom_pipe_height = self.height - (top_pipe_height + PIPE_GAP)

        wobble = random.randint(-PIPE_WOBBLE, PIPE_WOBBLE)

        x_center = self.width + PIPE_WIDTH // 2 + wobble  # За экраном справа + смещение

        # 2. Создаем графические объекты

        photo_top, photo_bottom = None, None
        top_id, bottom_id = None, None

        if self.pipe_top_original_img and self.pipe_bottom_original_img:
            # Масштабируем изображения под требуемую высоту
            img_top_resized = self.pipe_top_original_img.resize((PIPE_WIDTH, top_pipe_height))
            img_bottom_resized = self.pipe_bottom_original_img.resize((PIPE_WIDTH, bottom_pipe_height))

            # Создаем PhotoImage и сохраняем его в переменной
            photo_top = ImageTk.PhotoImage(img_top_resized)
            photo_bottom = ImageTk.PhotoImage(img_bottom_resized)

            # Верхняя труба (anchor='s' - якорь внизу, координаты x, y)
            top_id = self.canvas.create_image(
                x_center, top_pipe_height, image=photo_top, anchor='s'
            )

            # Нижняя труба (anchor='n' - якорь вверху, координаты x, y)
            bottom_y_start = top_pipe_height + PIPE_GAP
            bottom_id = self.canvas.create_image(
                x_center, bottom_y_start, image=photo_bottom, anchor='n'
            )
        else:
            # Заглушки-прямоугольники (как было)
            x_left = x_center - PIPE_WIDTH // 2
            x_right = x_center + PIPE_WIDTH // 2

            top_id = self.canvas.create_rectangle(
                x_left, 0, x_right, top_pipe_height, fill="green"
            )
            bottom_y_start = top_pipe_height + PIPE_GAP
            bottom_id = self.canvas.create_rectangle(
                x_left, bottom_y_start, x_right, self.height, fill="green"
            )

        # 3. Сохраняем в списке
        # Сохраняем PhotoImage, чтобы его не удалил сборщик мусора!
        self.pipes.append((top_id, bottom_id, x_center, photo_top, photo_bottom))

    def check_collisions(self):
        """Проверка столкновений с трубами (по фактическим координатам canvas)."""

        bird_x = 50
        bird_left, bird_right = bird_x - BIRD_RADIUS, bird_x + BIRD_RADIUS
        bird_top, bird_bottom = self.bird_y - BIRD_RADIUS, self.bird_y + BIRD_RADIUS

        # Итерируемся по кортежу (id_top, id_bottom, x_center, photo_top, photo_bottom)
        for pipe_data in self.pipes:
            top_id, bottom_id = pipe_data[0], pipe_data[1]

            pipe_coords_top = self.canvas.coords(top_id)
            pipe_coords_bottom = self.canvas.coords(bottom_id)

            if not pipe_coords_top or not pipe_coords_bottom:
                continue

            # Координаты трубы:
            # Для create_rectangle: (x1, y1, x2, y2)
            # Для create_image: (x_center, y_center)

            # Определяем горизонтальные границы трубы (зависит от того, что использовалось для отрисовки)
            if pipe_data[3] is not None:  # Если использовалось create_image
                # x_center - PIPE_WIDTH/2, x_center + PIPE_WIDTH/2
                x_center = pipe_coords_top[0]
                pipe_left = x_center - PIPE_WIDTH // 2
                pipe_right = x_center + PIPE_WIDTH // 2
            else:  # Если использовалось create_rectangle
                pipe_left = pipe_coords_top[0]
                pipe_right = pipe_coords_top[2]

            # Горизонтальное пересечение
            if bird_right > pipe_left and bird_left < pipe_right:

                # Коллизия с верхней трубой (Y-диапазон)
                # Для create_rectangle, y2 - 4-я координата
                # Для create_image (anchor='s'), y2 - 2-я координата (y_center)
                top_pipe_y2 = pipe_coords_top[1] if pipe_data[3] is not None else pipe_coords_top[3]

                if bird_top < top_pipe_y2:
                    return True

                # Коллизия с нижней трубой (Y-диапазон)
                # Для create_rectangle, y1 - 2-я координата
                # Для create_image (anchor='n'), y1 - 2-я координата (y_center)
                bottom_pipe_y1 = pipe_coords_bottom[1]  # Это y_center или y1

                if bird_bottom > bottom_pipe_y1:
                    return True

        return False

    def update_graphics(self):
        # Обновляем координаты птицы
        if self.bird_photo:
            self.canvas.coords(self.bird_obj, 50, self.bird_y)
        else:
            self.canvas.coords(
                self.bird_obj,
                50 - BIRD_RADIUS, self.bird_y - BIRD_RADIUS,
                50 + BIRD_RADIUS, self.bird_y + BIRD_RADIUS
            )

    # -----------------------
    # Input handlers & External API (без изменений)
    # -----------------------
    def apply_jump(self):
        self.bird_vy = JUMP_STRENGTH

    def on_jump(self, event=None):
        if self.running:
            self.apply_jump()

    def set_control_action(self, action: str):
        if action == 'jump' and self.running:
            self.external_jump_signal = True

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
        # print(f"Starting {self.__name__} camera analysis...")
        print(self.width)
        cfg.main_process_loger.info("Starting Flappy-bird camera analysis...")
        cfg.CameraProcessor.EYE_ANALYSIS=False
        cfg.CameraProcessor.HAND_DETECTION=False
        frame_count = 0
        cond= True
        cfg.CameraProcessor.EYE_ANALYSIS=False
        # launching = False
        while self.cond_video_process:
            if not cfg.CameraProcessor.last_command_executed:
                print("Waiting for last command to be executed...")
                is_cliped = not cfg.CameraProcessor.is_cliped
                if is_cliped and self.running:
                    print("Eyes are closed - jumping!")
                    cfg.main_process_loger.info("Flappy-bird: Eyes are closed - jumping!")
                    self.on_jump()
                cfg.CameraProcessor.last_command_executed = True
                is_cliped = False
                cfg.CameraProcessor.is_cliped = True
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
    try:
        root = tk.Tk()
        game = flappy_bird(root)
        root.mainloop()
    except Exception as e:
        print(f"Произошла ошибка при запуске Flappy Bird: {e}")