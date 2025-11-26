import tkinter as tk
import time
import random
import asyncio
from PIL import Image, ImageTk
from pkgs import functions
import global_config as cfg

# ==================================
# Глобальные переменные (Placeholder)
# ==================================
BACKGROUND_IMG = "app_imgs/labyrinth_bg.png"
TRACK_IMG = "app_imgs/labyrinth_wall.png"  # Бордюр трассы
PLAYER_IMG = "app_imgs/labyrinth_player.png"
FINISH_COLOR = "green"

TRACK_COLOR = "black"
PLAYER_COLOR = "red"
GRASS_COLOR = "#2E8B57"  # Зеленый для газона

# ==================================
# Параметры игры
# ==================================
CELL_SIZE = 40  # Уменьшаем для более плавных поворотов
TRACK_WIDTH = 120  # Ширина трассы
BORDER_WIDTH = 20  # Ширина бордюра
PLAYER_WIDTH = 30   # Ширина прямоугольного игрока
PLAYER_HEIGHT = 15  # Высота прямоугольного игрока
PLAYER_SPEED = 5
GAME_LEVEL = 1
TOTAL_LEVELS = 3
TOTAL = 6
VIEW_RADIUS = 800

# --- Параметры Тумана Войны ---
FOG_COLOR = "black"
FOG_STIPPLE = ""

# ==================================
# ПАКЕТЫ ТРАСС (0: газон, 1: трасса, 2: старт, 3: финиш)
# ==================================

# Уровень 1: Простая трасса с плавными поворотами
TRACK_PACK_1 = [
    # Трасса 1: Овал
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 2, 0, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 0, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 3, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
    
    ],
    # Трасса 2: S-образная
    [
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,1,1,1,1,1,0],
        [0,0,0,0,0,0,0,0,1,1,1,1,1,1,0],
        [0,0,0,0,0,0,0,1,1,1,1,1,1,1,0],
        [0,0,0,0,0,0,1,1,1,1,1,1,1,0,0],
        [0,0,0,0,0,1,1,1,1,1,1,1,0,0,0],
        [0,2,1,1,1,1,1,1,1,1,1,0,0,0,0],
        [0,1,1,1,1,1,1,1,1,0,0,0,0,0,0],
        [0,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
        [0,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,1,1,1,1,3,0,0,0,0,0,0,0,0,0],
        [0,1,1,1,1,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    ]
]

# Уровень 2: Более сложные трассы
TRACK_PACK_2 = [
    # Трасса с шпилькой
    [
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
        [0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0],
        [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
        [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
        [0,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
        [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0],
        [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0],
        [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
        [0,1,1,1,1,1,1,1,1,1,1,3,1,1,1,1,0],
        [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    ]
]

# Уровень 3: Сложные трассы
TRACK_PACK_3 = [
    # Сложная трасса с множеством поворотов
    [
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0],
        [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0],
        [0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
        [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
        [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
        [0,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
        [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0],
        [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0],
        [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,1,3,1,1,1,0],
        [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    ]
]

# Сборник всех паков трасс
TRACK_LAYOUTS_PACK = {
    1: TRACK_PACK_1,
    2: TRACK_PACK_2,
    3: TRACK_PACK_3,
}


class FormulaOneGame(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Formula One Racing")
        self.attributes('-fullscreen', True)
        self.grab_set()
        self.focus_force()
        self.attributes('-topmost', True)
        cfg.pass_launcher_menu=True

        self.width = self.winfo_screenwidth()
        self.height = self.winfo_screenheight()

        self.canvas = tk.Canvas(self, width=self.width, height=self.height, bg=GRASS_COLOR, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # --- Состояние игры ---
        self.current_track = None
        self.player_obj = None
        self.player_x = 0
        self.player_y = 0
        self.player_angle = 0  # Угол поворота машины
        self.finish_rect = None
        self.running = False
        self.game_after_id = None
        self.level = GAME_LEVEL

        self.bg_image_tk = None
        self.track_image_tk = None
        self.player_image_tk = None
        self.bg_image_obj = None

        # --- Хранилище объектов трассы ---
        self.track_objects = []

        # --- Хранилище для тумана ---
        self.fog_rects = {
            'top': None,
            'bottom': None,
            'left': None,
            'right': None
        }

        # --- Управление камерой/смещением ---
        self.target_x = self.width // 2
        self.target_y = self.height // 2
        self.track_offset_x = 0
        self.track_offset_y = 0
        self.cond_video_process = True

        # Bindings
        self.bind("<Escape>", lambda e: self.close_and_restore())
        self.bind("<Motion>", self.on_mouse_move)

        self.load_images()
        # self.run_camera_analysis()

        self.show_start_screen()

    def load_images(self):
        """Загружает и подготавливает все игровые изображения."""
        try:
            # Фон (газон) - растягиваем на весь экран
            bg_img = Image.open(BACKGROUND_IMG)
            bg_img = bg_img.resize((self.width, self.height), Image.LANCZOS)
            self.bg_image_tk = ImageTk.PhotoImage(bg_img)

            # Бордюр трассы
            track_img = Image.open(TRACK_IMG)
            track_img = track_img.resize((CELL_SIZE, CELL_SIZE), Image.NEAREST)
            self.track_image_tk = ImageTk.PhotoImage(track_img)

            # Игрок (прямоугольная машина)
            player_img = Image.open(PLAYER_IMG)
            player_img = player_img.resize((PLAYER_WIDTH, PLAYER_HEIGHT), Image.NEAREST)
            self.player_image_tk = ImageTk.PhotoImage(player_img)

            print("Все изображения успешно загружены.")

        except Exception as e:
            print(f"Ошибка загрузки изображений: {e}")
            print("Убедитесь, что файлы лежат в 'app_imgs/' и 'pip install Pillow' выполнен.")
            print("Игра будет использовать стандартные цвета.")
            self.bg_image_tk = None
            self.track_image_tk = None
            self.player_image_tk = None

    # -----------------------
    # Screens and Cleanup
    # -----------------------
    def clear_canvas(self):
        """Очищает холст."""
        self.canvas.delete("all")
        self.player_obj = None
        self.finish_rect = None
        self.bg_image_obj = None
        self.fog_rects = {'top': None, 'bottom': None, 'left': None, 'right': None}
        self.track_objects = []
        self.track_offset_x = 0
        self.track_offset_y = 0

    def close_and_restore(self):
        """Корректно закрывает игру и восстанавливает лаунчер"""
        self.stop_game_loop()
        self.cond_video_process = False
        
        if hasattr(self, 'canvas'):
            self.clear_canvas()
            self.canvas.destroy()
        
        if hasattr(self, 'master'):
            self.master.deiconify()
            self.master.lift()
            self.master.attributes('-topmost', True)
            
        cfg.initial_params_to_class()
        cfg.pass_launcher_menu = False
        cfg.CameraProcessor.deque_get_size=cfg.degue_size
        
        cfg.main_process_loger.info("Formula One game closed, returning to launcher. \n" \
            f"Params: EYE_ANALYSIS={cfg.EYE_ANALYSIS}, \nHAND_DETECTION={cfg.HAND_DETECTION}, " \
            f"\nOK_SIGHN_DETECTION={cfg.OK_SIGHN_DETECTION}, \nVICTORY_TOGETHER_SIGHN_DETECTION={cfg.VICTORY_TOGETHER_SIGHN_DETECTION}" \
            f"pass_launcher_menu={cfg.pass_launcher_menu}")
        
        self.destroy()
        self.destroy_self()

    def destroy_self(self):
        """Очищает все атрибуты класса"""
        print("Deactivating...")
        self.running = False
        self.cond_video_process = False
        
        if hasattr(self, 'game_after_id') and self.game_after_id:
            self.after_cancel(self.game_after_id)
        
        attrs = list(self.__dict__.keys())
        for k in attrs:
            try:
                delattr(self, k)
            except:
                pass

    def show_start_screen(self):
        self.stop_game_loop()
        self.clear_canvas()

        if self.bg_image_tk:
            self.bg_image_obj = self.canvas.create_image(
                self.width // 2, self.height // 2,
                image=self.bg_image_tk
            )

        self.canvas.create_text(self.width // 2, self.height // 3, text="Formula One Racing", fill="white",
                                font=("Arial", 48, "bold"))
        rules = (
            "Правила:\n\n"
            f"1. Проведите машину от старта (желтый) до финиша (зеленый).\n"
            "2. Управляйте машиной, задавая целевую позицию (мышью/камерой).\n"
            "3. Двигайтесь только по трассе - не съезжайте на газон!\n"
            f"4. Всего {TOTAL} уровней.\n\n"
            "Нажмите OK, чтобы начать гонку."
        )
        self.canvas.create_text(self.width // 2, int(self.height * 0.55), text=rules, fill="white",
                                font=("Arial", 18), width=int(self.width * 0.7))
        ok_btn = tk.Button(self, text="OK", font=("Arial", 20), command=self.start_game)
        self.canvas.create_window(self.width // 2, int(self.height * 0.75), window=ok_btn)

    def show_win_screen(self, final=False):
        self.stop_game_loop()
        self.clear_canvas()

        if self.bg_image_tk:
            self.bg_image_obj = self.canvas.create_image(
                self.width // 2, self.height // 2,
                image=self.bg_image_tk
            )

        title = "ПОБЕДА! Гонка завершена!" if final else f"Уровень {self.level} пройден!"
        btn_text = "Следующий уровень" if not final else "Restart"
        btn_command = self.next_level if not final else self.start_game

        self.canvas.create_text(self.width // 2, self.height // 3, text=title, fill=FINISH_COLOR,
                                font=("Arial", 40))

        next_btn = tk.Button(self, text=btn_text, font=("Arial", 18), command=btn_command)
        exit_btn = tk.Button(self, text="Exit", font=("Arial", 18), command=self.close_and_restore)
        self.canvas.create_window(self.width // 2 - 100, int(self.height * 0.6), window=next_btn)
        self.canvas.create_window(self.width // 2 + 100, int(self.height * 0.6), window=exit_btn)

    # -----------------------
    # Game lifecycle
    # -----------------------
    def start_game(self):
        self.level = 1
        self.load_level()

    def next_level(self):
        self.level += 1
        if self.level > TOTAL:
            self.show_win_screen(final=True)
        elif self.level > TOTAL_LEVELS:
            new_track = self.generate_racetrack(20)
            self.current_track = new_track
            self.load_level()
        else:
            self.load_level()

    def load_level(self):
        """Загружает и отрисовывает текущую трассу."""
        self.stop_game_loop()
        self.clear_canvas()

        if self.bg_image_tk:
            self.bg_image_obj = self.canvas.create_image(
                self.width // 2, self.height // 2,
                image=self.bg_image_tk
            )

        if self.level in TRACK_LAYOUTS_PACK:
            track_pack = TRACK_LAYOUTS_PACK[self.level]
            self.current_track = random.choice(track_pack)
        else:
            print(f"Ошибка: Трасса для уровня {self.level} не найдена. Использую уровень 1.")
            self.current_track = self.generate_racetrack(20)

        start_logical_x, start_logical_y = self.draw_track()

        self.player_x = start_logical_x
        self.player_y = start_logical_y
        self.player_angle = 0
        self.target_x = self.width // 2
        self.target_y = self.height // 2

        self.track_offset_x = (self.width // 2) - self.player_x
        self.track_offset_y = (self.height // 2) - self.player_y

        for obj in self.track_objects:
            self.canvas.move(obj, self.track_offset_x, self.track_offset_y)

        self.create_fog_of_war()
        self.draw_player()
        self.update_fog_of_war()

        self.running = True
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

    def draw_track(self):
        """
        Рисует трассу в ЛОГИЧЕСКИХ координатах.
        Возвращает ЛОГИЧЕСКИЕ координаты старта.
        """
        start_logical_x, start_logical_y = 0, 0

        for r, row in enumerate(self.current_track):
            for c, cell in enumerate(row):
                x1 = c * CELL_SIZE
                y1 = r * CELL_SIZE
                x2 = x1 + CELL_SIZE
                y2 = y1 + CELL_SIZE

                fill_color = None
                obj = None

                if cell == 1:  # Трасса
                    fill_color = "#404040"  # Серый асфальт
                    obj = self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color, outline="")

                elif cell == 2:  # Старт
                    start_logical_x = x1 + CELL_SIZE // 2
                    start_logical_y = y1 + CELL_SIZE // 2
                    fill_color = "yellow"
                    obj = self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color, outline="")

                elif cell == 3:  # Финиш
                    fill_color = FINISH_COLOR
                    obj = self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color, outline="")
                    self.finish_rect = obj

                if obj:
                    self.track_objects.append(obj)

        return start_logical_x, start_logical_y

    def draw_player(self):
        """Рисует прямоугольную машину в центре экрана."""
        screen_x = self.width // 2
        screen_y = self.height // 2

        if self.player_image_tk:
            self.player_obj = self.canvas.create_image(
                screen_x, screen_y,
                image=self.player_image_tk
            )
        else:
            # Рисуем прямоугольную машину
            w2 = PLAYER_WIDTH // 2
            h2 = PLAYER_HEIGHT // 2
            self.player_obj = self.canvas.create_rectangle(
                screen_x - w2, screen_y - h2,
                screen_x + w2, screen_y + h2,
                fill=PLAYER_COLOR, outline="white", width=2
            )

    def schedule_next_frame(self):
        self.game_after_id = self.after(16, self.game_loop)

    def game_loop(self):
        if not self.running:
            return

        self.move_player()
        
        if self.player_obj:
            self.canvas.lift(self.player_obj)

        if self.check_win():
            self.show_win_screen(final=(self.level == TOTAL))
            return

        self.schedule_next_frame()

    def move_player(self):
        """Двигает МИР трассы, а не игрока."""
        player_screen_x = self.width // 2
        player_screen_y = self.height // 2

        dx = self.target_x - player_screen_x
        dy = self.target_y - player_screen_y

        dist = (dx ** 2 + dy ** 2) ** 0.5
        if dist <= PLAYER_SPEED:
            return

        speed_factor = min(1, PLAYER_SPEED / dist)
        move_x = dx * speed_factor
        move_y = dy * speed_factor

        old_logical_x = self.player_x
        old_logical_y = self.player_y

        new_logical_x = self.player_x + move_x
        if not self.check_track_collision(new_logical_x, self.player_y):
            self.player_x = new_logical_x

        new_logical_y = self.player_y + move_y
        if not self.check_track_collision(self.player_x, new_logical_y):
            self.player_y = new_logical_y

        world_move_x = old_logical_x - self.player_x
        world_move_y = old_logical_y - self.player_y

        if world_move_x == 0 and world_move_y == 0:
            return

        for obj in self.track_objects:
            self.canvas.move(obj, world_move_x, world_move_y)

        self.track_offset_x += world_move_x
        self.track_offset_y += world_move_y

    def check_track_collision(self, logical_x, logical_y):
        """
        Проверяет коллизию с границами трассы.
        Возвращает True если машина съехала с трассы.
        """
        maze_r = int(logical_y // CELL_SIZE)
        maze_c = int(logical_x // CELL_SIZE)

        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                r_check = maze_r + dr
                c_check = maze_c + dc

                if not (0 <= r_check < len(self.current_track) and 0 <= c_check < len(self.current_track[0])):
                    continue

                # Если ячейка - газон (0), проверяем коллизию
                if self.current_track[r_check][c_check] == 0:
                    wall_x1 = c_check * CELL_SIZE
                    wall_y1 = r_check * CELL_SIZE
                    wall_x2 = wall_x1 + CELL_SIZE
                    wall_y2 = wall_y1 + CELL_SIZE

                    closest_x = max(wall_x1, min(logical_x, wall_x2))
                    closest_y = max(wall_y1, min(logical_y, wall_y2))

                    distance_sq = (logical_x - closest_x) ** 2 + (logical_y - closest_y) ** 2

                    # Используем меньший радиус для более точного определения коллизий
                    collision_radius = min(PLAYER_WIDTH, PLAYER_HEIGHT) // 2
                    if distance_sq < collision_radius ** 2:
                        return True

        return False

    def check_win(self):
        """Проверяет, достиг ли игрок финиша."""
        if self.finish_rect is None:
            return False

        player_screen_x = self.width // 2
        player_screen_y = self.height // 2
        r = max(PLAYER_WIDTH, PLAYER_HEIGHT) // 2

        overlapping_items = self.canvas.find_overlapping(
            player_screen_x - r, player_screen_y - r,
            player_screen_x + r, player_screen_y + r
        )
        return self.finish_rect in overlapping_items

    # -----------------------
    # Input Handlers
    # -----------------------
    def on_mouse_move(self, event):
        self.set_target_position(event.x, event.y)

    # -----------------------
    # External API
    # -----------------------
    def set_target_position(self, x: int | float, y: int | float):
        self.target_x = max(0, min(self.width, int(x)))
        self.target_y = max(0, min(self.height, int(y)))

    def create_fog_of_war(self):
        """Создает 4 прямоугольника тумана."""
        common_options = {
            'fill': FOG_COLOR,
            'stipple': FOG_STIPPLE,
            'outline': ""
        }
        self.fog_rects['top'] = self.canvas.create_rectangle(0, 0, 0, 0, **common_options)
        self.fog_rects['bottom'] = self.canvas.create_rectangle(0, 0, 0, 0, **common_options)
        self.fog_rects['left'] = self.canvas.create_rectangle(0, 0, 0, 0, **common_options)
        self.fog_rects['right'] = self.canvas.create_rectangle(0, 0, 0, 0, **common_options)

    def update_fog_of_war(self):
        """Обновляет координаты тумана вокруг статичного игрока."""
        if self.fog_rects['top'] is None:
            return

        px = self.width // 2
        py = self.height // 2

        r = VIEW_RADIUS
        w = self.width
        h = self.height

        self.canvas.coords(self.fog_rects['top'], 0, 0, w, py - r)
        self.canvas.coords(self.fog_rects['bottom'], 0, py + r, w, h)
        self.canvas.coords(self.fog_rects['left'], 0, py - r, px - r, py + r)
        self.canvas.coords(self.fog_rects['right'], px + r, py - r, w, py + r)

    def generate_racetrack(self, size: int = 20) -> list:
        """
        Генерирует гоночную трассу с плавными изгибами.
        """
        if size < 10:
            size = 10
        if size % 2 == 0:
            size += 1

        # Создаем базовый овал
        track = [[0 for _ in range(size)] for _ in range(size)]
        
        # Рисуем овал
        center_x, center_y = size // 2, size // 2
        radius_x, radius_y = size // 3, size // 4
        
        for r in range(size):
            for c in range(size):
                dx = (c - center_x) / radius_x
                dy = (r - center_y) / radius_y
                if dx*dx + dy*dy <= 1.0:
                    track[r][c] = 1
        
        # Добавляем старт и финиш
        track[center_y][center_x - radius_x + 1] = 2  # Старт
        track[center_y][center_x + radius_x - 1] = 3  # Финиш
        
        return track

    def run_camera_analysis(self):
        print("Launching Formula One camera analysis...")
        functions.run_async(self, self.camera_analysis_async())
    
    async def camera_analysis_async(self):
        cfg.main_process_loger.info("Starting Formula One camera analysis...")
        frame_count = 0
        cfg.CameraProcessor.EYE_ANALYSIS=cfg.EYE_ANALYSIS
        cfg.CameraProcessor.HAND_DETECTION=False
        cfg.CameraProcessor.deque_get_size=1
        
        while self.cond_video_process:
            if not cfg.CameraProcessor.last_command_executed:
                x,y = cfg.CameraProcessor.x, cfg.CameraProcessor.y
                if x is not None and y is not None:
                    self.set_target_position(x,y)
                    cfg.CameraProcessor.last_command_executed = True
            frame_count += 1
            await asyncio.sleep(0.1)


# -----------------------
# Запуск
# -----------------------
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    game = FormulaOneGame(root)
    root.mainloop()