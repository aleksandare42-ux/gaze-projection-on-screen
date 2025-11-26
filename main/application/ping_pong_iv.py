import tkinter as tk
import time
import asyncio
import global_config as cfg
from pkgs import functions
from PIL import Image, ImageTk

# BACKGROUND_IMG = "app_imgs/Gemini_Generated_Image_dk2dscdk2dscdk2d.png"
# BALL_IMG = "app_imgs/Gemini_Generated_Image_o8uleao8uleao8ul.png"
BALL_IMG = "app_imgs/pin_pong_ball.png"
BACKGROUND_IMG = "app_imgs/pin_pong_bg.png"

class PongGame(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Pong Game")
        self.attributes('-fullscreen', True)  # полноэкранный режим
        self.grab_set()       # Фокус и блокировка событий для других окон
        self.focus_force()    # Сразу в фокус
        self.attributes('-topmost', True)  # Поверх всех окон
        self.width = self.winfo_screenwidth()
        self.height = self.winfo_screenheight()
        self.cond_video_process = True

        # Canvas
        self.canvas = tk.Canvas(self, width=self.width, height=self.height, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # Game params
        self.paddle_width = int(self.width * 0.18)
        self.paddle_height = 20
        self.paddle_y = self.height - 60
        self.paddle_speed = 40  # скорость при нажатии клавиш
        self.ball_radius = 32
        self.ball_speed_x = 12
        self.ball_speed_y = -12

        # --- фон (через PIL) ---
        bg = Image.open(BACKGROUND_IMG)     # ← твой фон (можно PNG или JPG)
        bg = bg.resize((self.width, self.height))  # растягиваем под экран
        self.bg_photo = ImageTk.PhotoImage(bg)

        self.ball_spin_frames = []  # Список для кадров анимации
        self.ball_rotation_speed = 0
        self.current_spin_frame_index = 0.0

        try:
            ball_diameter = self.ball_radius * 2
            # 1. Загружаем ОРИГИНАЛ
            original_pil_ball = Image.open(BALL_IMG).resize((ball_diameter, ball_diameter))

            # 2. Создаем 12 кадров (каждые 30 градусов)
            # Можешь поменять 30 на 20 или 15 для более плавного вращения
            for angle in range(0, 360, 30):
                # .rotate() поворачивает ПРОТИВ часовой стрелки
                # expand=True важно, чтобы углы не обрезались
                rotated_img = original_pil_ball.rotate(angle, expand=True)
                self.ball_spin_frames.append(ImageTk.PhotoImage(rotated_img))

            print(f"--- Загружено {len(self.ball_spin_frames)} кадров анимации мяча ---")
            # self.ball_photo нам больше не нужна, у нас есть список
            self.ball_photo = self.ball_spin_frames[0]  # для совместимости

        except FileNotFoundError:
            print(f"!!! ОШИБКА: Не могу найти файл мяча по пути: {BALL_IMG}")
            self.ball_photo = None
        except Exception as e:
            print(f"!!! ОШИБКА при загрузке мяча: {e}")
            self.ball_photo = None




        # State
        self.paddle_x = (self.width // 2)
        self.external_paddle_x = None  # если задано извне - будет использоваться
        self.score = 0
        self.running = False
        self.game_after_id = None

        # Bindings
        self.bind("<Left>", self.on_left)
        self.bind("<Right>", self.on_right)
        self.bind("<Escape>", lambda e: self.close_and_restore())
        self.bind("<Key>", self.on_key)  # для возможности расширения

        # Create start and score frames (canvas items will be managed)
        self.start_items = []
        self.score_items = []

        # Create shapes placeholders
        self.paddle = None
        self.ball = None

        self.x_rocket_position = None

        # Show start screen
        self.show_start_screen()
        self.run_camera_analysis()

    # -----------------------
    # Screens
    # -----------------------
    def clear_canvas(self):
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.bg_photo, anchor="nw")
        self.start_items = []
        self.score_items = []
        self.paddle = None
        self.ball = None

    def close_and_restore(self):
        self.destroy()
        self.master.deiconify()                     # Показать снова
        self.master.lift()                          # Поднять на верх
        self.master.attributes('-topmost', True)    # Временно сделать поверх всех
        self.cond_video_process = False
        cfg.initial_params_to_class()
        cfg.pass_launcher_menu = False
        cfg.main_process_loger.info("Pong game closed, returning to launcher. \n" \
        f"Params: EYE_ANALYSIS={cfg.EYE_ANALYSIS}, \nHAND_DETECTION={cfg.HAND_DETECTION}, " \
        f"\nOK_SIGHN_DETECTION={cfg.OK_SIGHN_DETECTION}, \nVICTORY_TOGETHER_SIGHN_DETECTION={cfg.VICTORY_TOGETHER_SIGHN_DETECTION}"\
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
        title = "Пинг-Понг"
        rules = (
            "Правила:\n\n"
            "1. Управление: стрелки Влево/Вправо по умолчанию.\n"
            "2. Вы также можете программно задать позицию ракетки вызовом:\n"
            "   game.set_paddle_x(x)  # x — координата центра в пикселях\n"
            "3. Мяч отскакивает от стен и от ракетки.\n"
            "4. Если мяч коснулся нижней линии — игра закончится и вы увидите счёт.\n"
            "5. Нажмите ESC в любой момент чтобы вернуться на этот экран.\n\n"
            "Нажмите OK чтобы начать игру."
        )
        # Draw background text
        self.start_items.append(self.canvas.create_text(self.width//2, self.height//6, text=title, fill="white",
                                                        font=("Arial", 48, "bold")))
        self.start_items.append(self.canvas.create_text(self.width//2, self.height//3, text=rules, fill="white",
                                                        font=("Arial", 20), width=int(self.width*0.6)))
        ok_btn = tk.Button(self, text="OK", font=("Arial", 20), command=self.start_game)
        # place button using window_create so it sits on canvas
        btn_window = self.canvas.create_window(self.width//2, int(self.height*0.75), window=ok_btn)
        self.start_items.append(btn_window)

    def show_score_screen(self):
        self.stop_game_loop()
        self.clear_canvas()
        text = f"Игра окончена.\nСчёт (отбитых мячей): {self.score}"
        self.score_items.append(self.canvas.create_text(self.width//2, self.height//3, text=text, fill="white",
                                                        font=("Arial", 40)))
        restart_btn = tk.Button(self, text="Restart", font=("Arial", 18), command=self.start_game)
        exit_btn = tk.Button(self, text="Exit", font=("Arial", 18), command=self.destroy)
        self.score_items.append(self.canvas.create_window(self.width//2 - 100, int(self.height*0.6), window=restart_btn))
        self.score_items.append(self.canvas.create_window(self.width//2 + 100, int(self.height*0.6), window=exit_btn))

    # -----------------------
    # Game lifecycle
    # -----------------------
    def start_game(self):
        self.clear_canvas()
        self.score = 0
        self.ball_rotation_speed = 0
        self.current_spin_frame_index = 0.0
        self.running = True

        # Initial paddle & ball positions
        self.paddle_x = self.width // 2
        self.paddle = self.canvas.create_rectangle(0, 0, 0, 0, fill="white")
        self.ball_x = self.width // 2
        self.ball_y = self.height // 2
        self.ball_vx = self.ball_speed_x
        self.ball_vy = self.ball_speed_y
        # Проверяем, что у нас есть список кадров
        if self.ball_spin_frames:
            # Если да - создаем мяч-картинку
            self.ball = self.canvas.create_image(self.ball_x, self.ball_y, image=self.ball_spin_frames[0])
        else:
            # Если нет (картинки не загрузились) - создаем запасной мяч-овал
            self.ball = self.canvas.create_oval(0, 0, 0, 0, fill="white")
        # Draw initially
        self.draw_paddle()
        self.draw_ball()

        # Start loop
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
    def draw_paddle(self):
        left = self.paddle_x - self.paddle_width // 2
        right = self.paddle_x + self.paddle_width // 2
        top = self.paddle_y - self.paddle_height // 2
        bottom = self.paddle_y + self.paddle_height // 2
        # clamp
        if left < 0:
            left = 0
            right = self.paddle_width
            self.paddle_x = right - self.paddle_width//2
        if right > self.width:
            right = self.width
            left = self.width - self.paddle_width
            self.paddle_x = left + self.paddle_width//2
        self.canvas.coords(self.paddle, left, top, right, bottom)

    # def draw_ball(self):
    #     # вместо oval — просто двигаем изображение
    #     self.canvas.coords(self.ball, self.ball_x-self.ball_radius//2, self.ball_y- self.ball_radius//2,
    #                        self.ball_x + self.ball_radius//2, self.ball_y + self.ball_radius//2)
    def draw_ball(self):
        # Если у нас есть кадры вращения (т.е. картинки загрузились)
        if self.ball_spin_frames:
            # Получаем ЦЕЛЫЙ индекс кадра
            frame_index = int(self.current_spin_frame_index)

            # Меняем картинку на нужный кадр
            try:
                self.canvas.itemconfig(self.ball, image=self.ball_spin_frames[frame_index])
            except tk.TclError:
                # Окно могло быть уже закрыто, ничего страшного
                pass

            # И двигаем ее (картинка двигается по центру, 2-мя координатами)
            self.canvas.coords(self.ball, self.ball_x, self.ball_y)

        else:
            # Старая логика для овала (если картинки НЕ загрузились)
            # (Этот код будет работать, т.к. start_game тоже войдет в 'else'
            # и создаст self.ball как овал, который ждет 4 koord.)
            r = self.ball_radius
            self.canvas.coords(self.ball, self.ball_x - r, self.ball_y - r, self.ball_x + r, self.ball_y + r)


    def schedule_next_frame(self):
        # 60 FPS ~ 16 ms
        self.game_after_id = self.after(16, self.game_loop)

    def game_loop(self):
        if not self.running:
            return

            # --- Обновление вращения ---
        if self.ball_spin_frames:  # Вращаем, только если есть кадры
            # 1. Применяем затухание (трение воздуха)
            self.ball_rotation_speed *= 0.985  # Замедляется на 1.5% каждый кадр
            if abs(self.ball_rotation_speed) < 0.05:
                self.ball_rotation_speed = 0  # Остановка, если почти не крутится

            # 2. Обновляем индекс кадра
            num_frames = len(self.ball_spin_frames)
            # Прибавляем скорость. self.current_spin_frame_index - это float
            self.current_spin_frame_index = (self.current_spin_frame_index + self.ball_rotation_speed) % num_frames

        # If external paddle pos provided, use it (center x)
        if self.external_paddle_x is not None:
            # clamp external value
            x = int(self.external_paddle_x)
            if x < self.paddle_width//2:
                x = self.paddle_width//2
            if x > self.width - self.paddle_width//2:
                x = self.width - self.paddle_width//2
            self.paddle_x = x

        # Move ball
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # Wall collisions (left/right)
        if self.ball_x - self.ball_radius <= 0:
            self.ball_x = self.ball_radius
            self.ball_vx = -self.ball_vx
        if self.ball_x + self.ball_radius >= self.width:
            self.ball_x = self.width - self.ball_radius
            self.ball_vx = -self.ball_vx

        # Top collision
        if self.ball_y - self.ball_radius <= 0:
            self.ball_y = self.ball_radius
            self.ball_vy = -self.ball_vy

        # Paddle collision
        paddle_left = self.paddle_x - self.paddle_width//2
        paddle_right = self.paddle_x + self.paddle_width//2
        paddle_top = self.paddle_y - self.paddle_height//2

        # Simple AABB check when ball is descending
        if self.ball_vy > 0 and (self.ball_y + self.ball_radius) >= paddle_top and \
           (paddle_left <= self.ball_x <= paddle_right) and (self.ball_y - self.ball_radius) < (self.paddle_y + self.paddle_height):
            # Bounce
            self.ball_y = paddle_top - self.ball_radius
            self.ball_vy = -abs(self.ball_vy)
            # tweak horizontal velocity depending on where it hit the paddle
            hit_pos = (self.ball_x - self.paddle_x) / (self.paddle_width / 2)  # -1 .. 1
            self.ball_vx += hit_pos * 4  # add spin
            # limit vx
            max_vx = 18
            if self.ball_vx > max_vx: self.ball_vx = max_vx
            if self.ball_vx < -max_vx: self.ball_vx = -max_vx
            MAX_ROTATION_SPEED = 1.8  # (попробуй от 1.0 до 3.0)
            self.ball_rotation_speed = hit_pos * MAX_ROTATION_SPEED
            self.score += 1

        # Bottom miss -> end round
        if self.ball_y - self.ball_radius > self.height:
            # show score screen
            self.show_score_screen()
            return

        # Update graphics
        self.draw_ball()
        self.draw_paddle()

        # Continue
        self.schedule_next_frame()

    # -----------------------
    # Input handlers
    # -----------------------
    def on_left(self, event=None):
        # если внешнее управление не задано, двигаем клавишами
        if self.external_paddle_x is None:
            self.paddle_x -= self.paddle_speed
            if self.paddle_x < self.paddle_width//2:
                self.paddle_x = self.paddle_width//2
            self.draw_paddle()

    def on_right(self, event=None):
        if self.external_paddle_x is None:
            self.paddle_x += self.paddle_speed
            if self.paddle_x > self.width - self.paddle_width//2:
                self.paddle_x = self.width - self.paddle_width//2
            self.draw_paddle()

    def on_escape(self, event=None):
        # ESC -> всегда на стартовый экран
        self.show_start_screen()

    def on_key(self, event):
        # можно расширить управляющие клавиши
        pass

    # -----------------------
    # External API
    # -----------------------
    def set_paddle_x(self, x: int | float | None):
        """
        Установить позицию ракетки (центр) в экранных пикселях.
        Если передать None — возвращается управление клавишам.
        """
        if x is None:
            self.external_paddle_x = None
        else:
            try:
                self.external_paddle_x = int(x)
            except Exception:
                self.external_paddle_x = None

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
        print("Starting ping_pong camera analysis...")
        print(self.width)
        cfg.main_process_loger.info("Starting ping_pong camera analysis...")
        frame_count = 0
        cond= True
        cfg.CameraProcessor.EYE_ANALYSIS=False
        # launching = False
        while self.cond_video_process:
            if cfg.VICTORY_TOGETHER_SIGHN_DETECTION:
                x, y, = cfg.CameraProcessor.x, cfg.CameraProcessor.y
                if x is not None and y is not None and not cfg.CameraProcessor.last_command_executed:
                    # x*=self.width
                    print("Camera x,y:", x, y, cfg.CameraProcessor.last_command_executed)
                    if self.x_rocket_position is not None:
                        delta_x = x - self.x_rocket_position
                        delta_x*=1.5
                        self.x_rocket_position+=delta_x
                        self.paddle_x+=delta_x
                        if self.paddle_x < self.paddle_width//2:
                            self.paddle_x = self.paddle_width//2
                        if self.paddle_x > self.width - self.paddle_width//2:
                            self.paddle_x = self.width - self.paddle_width//2
                        print("Trying to move rocket to:", self.x_rocket_position)
                        self.draw_paddle()
                    else:
                        self.x_rocket_position = self.paddle_x
                    cfg.CameraProcessor.last_command_executed = True
                else: self.x_rocket_position=None
            else: self.x_rocket_position=None
            frame_count += 1
            await asyncio.sleep(0.05)

# -----------------------
# Запуск
# -----------------------
if __name__ == "__main__":
    root = tk.Tk()
    game = PongGame(root)

    # Пример: если вы хотите управлять ракеткой из вне — можно сделать так:
    # (раскомментируйте и подайте координаты)
    #
    # def follow_mouse(event):
    #     game.set_paddle_x(event.x)
    # root.bind("<Motion>", follow_mouse)
    #
    # Или программно:
    # root.after(2000, lambda: game.set_paddle_x(200))  # через 2 сек поставит центр ракетки в x=200

    root.mainloop()
