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
WALL_IMG = "app_imgs/labyrinth_wall.png"
PLAYER_IMG = "app_imgs/labyrinth_player.png"
FINISH_COLOR = "green"

WALL_COLOR = "black"
PLAYER_COLOR = "red"

# ==================================
# Параметры игры
# ==================================
CELL_SIZE = 500  # Размер одной ячейки (блока) в пикселях
# PLAYER_RADIUS = 10
# PLAYER_SPEED = 5
GAME_LEVEL = 1
TOTAL_LEVELS = 3  # Общее количество уровней
TOTAL = 6
PLAYER_RADIUS = max(12, CELL_SIZE // 3)   # радиус игрока в пикселях
PLAYER_SPEED = 30 # max(4, CELL_SIZE // 10)    # скорость (больше CELL_SIZE -> чуть больше скорости)
VIEW_RADIUS = 800 # max(48, CELL_SIZE * 2 // 3)

# --- Параметры Тумана Войны ---
# VIEW_RADIUS = 75  # Видимый радиус вокруг игрока в пикселях
FOG_COLOR = "black"  # Цвет тумана
FOG_STIPPLE = ""  # "gray50" для полупрозрачности

# ==================================
# ПАКЕТЫ ЛАБИРИНТОВ (0: путь, 1: стена, 2: старт, 3: финиш)
# ==================================

# Уровень 1 (Пример: 9x9)
MAZE_PACK_1 = [
    # Карта 1: Простой
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
    # Карта 2: С поворотом
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 2, 1, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 1, 3, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]
]

# Уровень 2 (Пример: 11x11)
MAZE_PACK_2 = [
    # Карта 1: Средний
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 2, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]
]

# Уровень 3 (Пример: 13x13)
MAZE_PACK_3 = [
    # Карта 1: Сложный
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 2, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]
]

# Сборник всех паков (для индексации по уровню)
MAZE_LAYOUTS_PACK = {
    1: MAZE_PACK_1,
    2: MAZE_PACK_2,
    3: MAZE_PACK_3,
}


class LabyrinthGame(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Labyrinth Game")
        self.attributes('-fullscreen', True)
        self.grab_set()
        self.focus_force()
        self.attributes('-topmost', True)
        cfg.pass_launcher_menu=True

        self.width = self.winfo_screenwidth()
        self.height = self.winfo_screenheight()

        self.canvas = tk.Canvas(self, width=self.width, height=self.height, bg="#333333", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # --- Состояние игры ---
        self.current_maze = None
        self.player_obj = None  # Игрок (объект) - теперь статичен
        self.player_x = 0  # ЛОГИЧЕСКАЯ X позиция игрока
        self.player_y = 0  # ЛОГИЧЕСКАЯ Y позиция игрока
        self.finish_rect = None
        self.running = False
        self.game_after_id = None
        self.level = GAME_LEVEL

        self.bg_image_tk = None
        self.wall_image_tk = None
        self.player_image_tk = None
        self.bg_image_obj = None

        # --- НОВОЕ: Хранилище объектов лабиринта ---
        self.maze_objects = []

        # --- Хранилище для 4-х прямоугольников тумана ---
        self.fog_rects = {
            'top': None,
            'bottom': None,
            'left': None,
            'right': None
        }

        # --- Управление камерой/смещением ---
        self.target_x = self.width // 2  # Цель (курсор)
        self.target_y = self.height // 2  # Цель (курсор)
        self.maze_offset_x = 0  # Смещение лабиринта X
        self.maze_offset_y = 0  # Смещение лабиринта Y
        self.cond_video_process = True

        # Bindings
        self.bind("<Escape>", lambda e: self.close_and_restore())
        self.bind("<Motion>", self.on_mouse_move)

        self.load_images()
        self.run_camera_analysis()

        self.show_start_screen()

    def load_images(self):
        """Загружает и подготавливает все игровые изображения."""
        try:
            # 1. Фон (растягиваем на весь экран)
            bg_img = Image.open(BACKGROUND_IMG)
            bg_img = bg_img.resize((self.width, self.height), Image.LANCZOS)
            self.bg_image_tk = ImageTk.PhotoImage(bg_img)

            # 2. Стена (размер ячейки, NEAREST для пиксель-арта)
            wall_img = Image.open(WALL_IMG)
            wall_img = wall_img.resize((CELL_SIZE, CELL_SIZE), Image.NEAREST)
            self.wall_image_tk = ImageTk.PhotoImage(wall_img)

            # 3. Игрок (размер по радиусу, NEAREST для пиксель-арта)
            player_size = int(PLAYER_RADIUS * 2)
            player_img = Image.open(PLAYER_IMG)
            player_img = player_img.resize((player_size, player_size), Image.NEAREST)
            self.player_image_tk = ImageTk.PhotoImage(player_img)

            print("Все изображения успешно загружены.")

        except Exception as e:
            print(f"Ошибка загрузки изображений: {e}")
            print("Убедитесь, что файлы лежат в 'app_imgs/' и 'pip install Pillow' выполнен.")
            print("Игра будет использовать стандартные цвета.")
            self.bg_image_tk = None
            self.wall_image_tk = None
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
        # --- НОВОЕ: Сброс тумана и объектов лабиринта ---
        self.fog_rects = {'top': None, 'bottom': None, 'left': None, 'right': None}
        self.maze_objects = []
        self.maze_offset_x = 0
        self.maze_offset_y = 0

    def close_and_restore(self):
        """Корректно закрывает игру и восстанавливает лаунчер"""
        self.stop_game_loop()  # останавливаем игровой цикл
        self.cond_video_process = False  # останавливаем обработку камеры
        
        # Очищаем и уничтожаем канвас
        if hasattr(self, 'canvas'):
            self.clear_canvas()
            self.canvas.destroy()
        
        # Восстанавливаем лаунчер
        if hasattr(self, 'master'):
            self.master.deiconify()
            self.master.lift()
            self.master.attributes('-topmost', True)
            
        # Сбрасываем параметры конфига
        cfg.initial_params_to_class()
        cfg.pass_launcher_menu = False
        cfg.CameraProcessor.deque_get_size=cfg.degue_size
        
        # Логируем закрытие
        cfg.main_process_loger.info("Labyrinth game closed, returning to launcher. \n" \
            f"Params: EYE_ANALYSIS={cfg.EYE_ANALYSIS}, \nHAND_DETECTION={cfg.HAND_DETECTION}, " \
            f"\nOK_SIGHN_DETECTION={cfg.OK_SIGHN_DETECTION}, \nVICTORY_TOGETHER_SIGHN_DETECTION={cfg.VICTORY_TOGETHER_SIGHN_DETECTION}" \
            f"pass_launcher_menu={cfg.pass_launcher_menu}")
        
        # Уничтожаем объекты игры
        self.destroy()
        self.destroy_self()

    def destroy_self(self):
        """Очищает все атрибуты класса"""
        print("Deactivating...")
        # Останавливаем все циклы
        self.running = False
        self.cond_video_process = False
        
        # Отменяем все отложенные задачи
        if hasattr(self, 'game_after_id') and self.game_after_id:
            self.after_cancel(self.game_after_id)
        
        # Очищаем все атрибуты
        attrs = list(self.__dict__.keys())
        for k in attrs:
            try:
                delattr(self, k)
            except:
                pass

    def show_start_screen(self):
        # ... (Код БЕЗ ИЗМЕНЕНИЙ) ...
        self.stop_game_loop()
        self.clear_canvas()

        if self.bg_image_tk:
            self.bg_image_obj = self.canvas.create_image(
                self.width // 2, self.height // 2,
                image=self.bg_image_tk
            )

        self.canvas.create_text(self.width // 2, self.height // 3, text="Лабиринт", fill=WALL_COLOR,
                                font=("Arial", 48, "bold"))
        rules = (
            "Правила:\n\n"
            f"1. Проведите игрока от старта (желтый) до финиша (зеленый).\n"
            "2. Вы управляете игроком, задавая ему целевую позицию (мышью/камерой).\n"
            "3. Нельзя проходить сквозь стены.\n"
            f"4. Всего {TOTAL} уровней.\n\n"
            "Нажмите OK, чтобы начать игру."
        )
        self.canvas.create_text(self.width // 2, int(self.height * 0.55), text=rules, fill=WALL_COLOR,
                                font=("Arial", 18), width=int(self.width * 0.7))
        ok_btn = tk.Button(self, text="OK", font=("Arial", 20), command=self.start_game)
        self.canvas.create_window(self.width // 2, int(self.height * 0.75), window=ok_btn)

    def show_win_screen(self, final=False):
        # ... (Код БЕЗ ИЗМЕНЕНИЙ) ...
        self.stop_game_loop()
        self.clear_canvas()

        if self.bg_image_tk:
            self.bg_image_obj = self.canvas.create_image(
                self.width // 2, self.height // 2,
                image=self.bg_image_tk
            )

        title = "ПОБЕДА!" if final else f"Уровень {self.level} пройден!"
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
            # Сгенерировать новый случайный уровень 15x15, пометить старт/финиш
            new_maze = self.generate_random_maze(15)
            self.current_maze = new_maze
            # Сбросить уровень и загрузить созданную карту
            # self.level = 1
            self.load_level()
        else:
            self.load_level()

    # --- ИЗМЕНЕН МЕТОД ---
    def load_level(self):
        """Загружает и отрисовывает текущий лабиринт, выбирая его случайным образом."""
        self.stop_game_loop()
        self.clear_canvas()  # self.maze_objects сбрасывается здесь

        # 1. Отрисовка фона
        if self.bg_image_tk:
            self.bg_image_obj = self.canvas.create_image(
                self.width // 2, self.height // 2,
                image=self.bg_image_tk
            )

        # 2. Случайный выбор лабиринта
        if self.level in MAZE_LAYOUTS_PACK:
            maze_pack = MAZE_LAYOUTS_PACK[self.level]
            self.current_maze = random.choice(maze_pack)
        else:
            print(f"Ошибка: Лабиринт для уровня {self.level} не найден. Использую уровень 1.")
            # self.current_maze = random.choice(MAZE_LAYOUTS_PACK[1])
            self.current_maze = self.generate_random_maze(15)

        # 3. Рисуем лабиринт в ЛОГИЧЕСКИХ координатах (0,0)
        #    и получаем ЛОГИЧЕСКУЮ позицию старта
        start_logical_x, start_logical_y = self.draw_maze()

        # 4. Устанавливаем ЛОГИЧЕСКУЮ позицию игрока
        self.player_x = start_logical_x
        self.player_y = start_logical_y
        self.target_x = self.width // 2  # Сбрасываем цель в центр
        self.target_y = self.height // 2

        # 5. Рассчитываем начальное смещение, чтобы старт был в центре
        self.maze_offset_x = (self.width // 2) - self.player_x
        self.maze_offset_y = (self.height // 2) - self.player_y

        # 6. Сдвигаем все объекты лабиринта на начальное смещение
        for obj in self.maze_objects:
            self.canvas.move(obj, self.maze_offset_x, self.maze_offset_y)

        # 7. Создаем туман (он не движется)
        self.create_fog_of_war()

        # 8. Рисуем игрока СТАТИЧНО в центре
        self.draw_player()

        # 9. Обновляем позицию тумана (один раз, т.к. игрок не движется)
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

    # --- ИЗМЕНЕН МЕТОД ---
    def draw_maze(self):
        """
        Рисует лабиринт в ЛОГИЧЕСКИХ координатах (вокруг 0,0)
        и сохраняет все объекты в self.maze_objects.
        Возвращает ЛОГИЧЕСКИЕ координаты старта.
        """
        start_logical_x, start_logical_y = 0, 0

        for r, row in enumerate(self.current_maze):
            for c, cell in enumerate(row):
                # Рисуем в логических координатах (без offset)
                x1 = c * CELL_SIZE
                y1 = r * CELL_SIZE
                x2 = x1 + CELL_SIZE
                y2 = y1 + CELL_SIZE

                fill_color = None
                obj = None

                if cell == 1:  # Стена
                    if self.wall_image_tk:
                        obj = self.canvas.create_image(x1, y1, image=self.wall_image_tk, anchor=tk.NW)
                    else:
                        obj = self.canvas.create_rectangle(x1, y1, x2, y2, fill=WALL_COLOR, outline="")

                elif cell == 2:  # Старт
                    start_logical_x = x1 + CELL_SIZE // 2
                    start_logical_y = y1 + CELL_SIZE // 2
                    fill_color = "yellow"
                    obj = self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color, outline="")

                elif cell == 3:  # Финиш
                    fill_color = FINISH_COLOR
                    obj = self.canvas.create_rectangle(x1, y1, x2, y2, fill=fill_color, outline="")
                    self.finish_rect = obj  # Сохраняем финиш отдельно для check_win

                # Добавляем созданный объект в список
                if obj:
                    self.maze_objects.append(obj)

        return start_logical_x, start_logical_y

    # --- ИЗМЕНЕН МЕТОД ---
    def draw_player(self):
        """Рисует игрока СТАТИЧНО в центре экрана."""

        # Экранные (статичные) координаты
        screen_x = self.width // 2
        screen_y = self.height // 2

        if self.player_image_tk:
            self.player_obj = self.canvas.create_image(
                screen_x, screen_y,
                image=self.player_image_tk
            )
        else:
            r = PLAYER_RADIUS
            self.player_obj = self.canvas.create_oval(
                screen_x - r, screen_y - r,
                screen_x + r, screen_y + r,
                fill=PLAYER_COLOR
            )

    def schedule_next_frame(self):
        # ~ 60 FPS
        self.game_after_id = self.after(16, self.game_loop)

    # --- ИЗМЕНЕН МЕТОД ---
    def game_loop(self):
        if not self.running:
            return

        # 1. Движение МИРА (логика в move_player)
        self.move_player()

        # 2. Обновление тумана (теперь не нужно, т.к. игрок статичен)
        # self.update_fog_of_war() # <-- Раскомментируйте, если VIEW_RADIUS должен меняться

        # 3. Поднять игрока над туманом и лабиринтом
        if self.player_obj:
            self.canvas.lift(self.player_obj)

        # 4. Проверка на победу
        if self.check_win():
            self.show_win_screen(final=(self.level == TOTAL))
            return

        # 5. Продолжение цикла
        self.schedule_next_frame()

    # --- ИЗМЕНЕН МЕТОД ---
    def move_player(self):
        """
        Теперь эта функция двигает МИР, а не игрока.
        Она обновляет ЛОГИЧЕСКИЕ self.player_x/y
        и двигает ВСЕ self.maze_objects.
        """

        # 1. Рассчитываем вектор движения от ЦЕНТРА ЭКРАНА к цели (курсору)
        player_screen_x = self.width // 2
        player_screen_y = self.height // 2

        dx = self.target_x - player_screen_x
        dy = self.target_y - player_screen_y

        dist = (dx ** 2 + dy ** 2) ** 0.5
        if dist <= PLAYER_SPEED:  # Если курсор близко к игроку, не двигаемся
            return

        speed_factor = min(1, PLAYER_SPEED / dist)
        move_x = dx * speed_factor
        move_y = dy * speed_factor

        # Сохраняем старую ЛОГИЧЕСКУЮ позицию
        old_logical_x = self.player_x
        old_logical_y = self.player_y

        # 2. Проверяем движение по X
        new_logical_x = self.player_x + move_x
        if not self.check_wall_collision(new_logical_x, self.player_y):
            self.player_x = new_logical_x

        # 3. Проверяем движение по Y
        new_logical_y = self.player_y + move_y
        if not self.check_wall_collision(self.player_x, new_logical_y):
            self.player_y = new_logical_y

        # 4. Рассчитываем, насколько мир должен сдвинуться (в обратную сторону)
        world_move_x = old_logical_x - self.player_x
        world_move_y = old_logical_y - self.player_y

        if world_move_x == 0 and world_move_y == 0:
            return  # Столкнулись со стеной в обе стороны

        # 5. Двигаем КАЖДЫЙ объект лабиринта
        for obj in self.maze_objects:
            self.canvas.move(obj, world_move_x, world_move_y)

        # 6. Обновляем глобальное смещение (для отладки или будущих нужд)
        self.maze_offset_x += world_move_x
        self.maze_offset_y += world_move_y

        # Игрок (self.player_obj) НЕ ДВИГАЕТСЯ

    # --- ИЗМЕНЕН МЕТОД ---
    def check_wall_collision(self, logical_x, logical_y):
        """
        Проверяет коллизию в ЛОГИЧЕСКИХ координатах.
        """
        r = PLAYER_RADIUS

        # 1. Находим ячейку лабиринта, в которой находится игрок
        maze_r = int(logical_y // CELL_SIZE)
        maze_c = int(logical_x // CELL_SIZE)

        # 2. Проверяем 9 ячеек вокруг (включая свою)
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                r_check = maze_r + dr
                c_check = maze_c + dc

                # Проверка выхода за границы массива
                if not (0 <= r_check < len(self.current_maze) and 0 <= c_check < len(self.current_maze[0])):
                    continue

                # Если ячейка - стена (1)
                if self.current_maze[r_check][c_check] == 1:

                    # 3. Находим ЛОГИЧЕСКИЕ координаты этой стены
                    wall_x1 = c_check * CELL_SIZE
                    wall_y1 = r_check * CELL_SIZE
                    wall_x2 = wall_x1 + CELL_SIZE
                    wall_y2 = wall_y1 + CELL_SIZE

                    # 4. Находим ближайшую точку на стене к ЛОГИЧЕСКОЙ позиции игрока
                    closest_x = max(wall_x1, min(logical_x, wall_x2))
                    closest_y = max(wall_y1, min(logical_y, wall_y2))

                    # 5. Проверяем расстояние от игрока до этой точки
                    distance_sq = (logical_x - closest_x) ** 2 + (logical_y - closest_y) ** 2

                    if distance_sq < r ** 2:
                        return True  # Коллизия

        return False  # Нет коллизии

    # --- ИЗМЕНЕН МЕТОД ---
    def check_win(self):
        """
        Проверяет, пересекается ли СТАТИЧНЫЙ игрок
        с ПОДВИЖНЫМ финишным прямоугольником.
        """
        if self.finish_rect is None:
            return False

        # Используем СТАТИЧНЫЕ экранные координаты игрока
        player_screen_x = self.width // 2
        player_screen_y = self.height // 2
        r = PLAYER_RADIUS

        overlapping_items = self.canvas.find_overlapping(
            player_screen_x - r, player_screen_y - r,
            player_screen_x + r, player_screen_y + r
        )
        return self.finish_rect in overlapping_items

    # -----------------------
    # Input Handlers
    # -----------------------
    def on_mouse_move(self, event):
        """Для отладки и управления по умолчанию."""
        self.set_target_position(event.x, event.y)

    # -----------------------
    # External API (Для управления через камеру)
    # -----------------------
    def set_target_position(self, x: int | float, y: int | float):
        """
        Устанавливает целевую позицию (курсор), к которой будет двигаться мир.
        """
        self.target_x = max(0, min(self.width, int(x)))
        self.target_y = max(0, min(self.height, int(y)))

    # --- НОВЫЙ МЕТОД ---
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

    # --- ИЗМЕНЕН МЕТОД ---
    def update_fog_of_war(self):
        """Обновляет координаты тумана вокруг СТАТИЧНОГО игрока."""
        if self.fog_rects['top'] is None:
            return  # Туман еще не создан

        # Используем СТАТИЧНЫЕ экранные координаты игрока
        px = self.width // 2
        py = self.height // 2

        r = VIEW_RADIUS
        w = self.width
        h = self.height

        # [Top]    Верхний прямоугольник: от (0,0) до (ширина, y_игрока - радиус)
        self.canvas.coords(self.fog_rects['top'], 0, 0, w, py - r)

        # [Bottom] Нижний прямоугольник: от (0, y_игрока + радиус) до (ширина, высота)
        self.canvas.coords(self.fog_rects['bottom'], 0, py + r, w, h)

        # [Left]   Левый прямоугольник: от (0, y_игрока - радиус) до (x_игрока - радиус, y_игрока + радиус)
        self.canvas.coords(self.fog_rects['left'], 0, py - r, px - r, py + r)

        # [Right]  Правый прямоугольник: от (x_игрока + радиус, y_игрока - радиус) до (ширина, y_игрока + радиус)
        self.canvas.coords(self.fog_rects['right'], px + r, py - r, w, py + r)

    def generate_random_maze(self, size: int = 15) -> list:
        """
        Генерирует лабиринт размера size x size, возвращает список списков:
        1 = стена, 0 = путь, 2 = старт, 3 = финиш.
        Алгоритм: рекурсивный backtracker на сетке с шагом 2 (требует нечетного size).
        Результат гарантированно содержит проходимый путь от старта к финишу.
        """
        if size < 5:
            size = 5
        if size % 2 == 0:
            size += 1

        # Инициализация заполненной стенами карты
        maze = [[1 for _ in range(size)] for _ in range(size)]

        # Начальная клетка для резьбы (используем нечётные индексы)
        start_r, start_c = 1, 1
        maze[start_r][start_c] = 0

        from random import shuffle, randrange
        stack = [(start_r, start_c)]
        directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]

        while stack:
            r, c = stack[-1]
            neigh = []
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 1 <= nr < size - 1 and 1 <= nc < size - 1 and maze[nr][nc] == 1:
                    neigh.append((nr, nc, dr, dc))
            if neigh:
                shuffle(neigh)
                nr, nc, dr, dc = neigh[0]
                # Вырываем стену между (r,c) и (nr,nc)
                wall_r, wall_c = r + dr // 2, c + dc // 2
                maze[wall_r][wall_c] = 0
                maze[nr][nc] = 0
                stack.append((nr, nc))
            else:
                stack.pop()

        # Установим старт и финиш на проходах: левый верх / правый низ (можно рандомизировать)
        # Поиск ближайшей проходной клетки к (1,1) и (size-2,size-2)
        def find_nearest_path(sr, sc):
            from collections import deque
            q = deque([(sr, sc)])
            seen = { (sr, sc) }
            while q:
                r, c = q.popleft()
                if 0 <= r < size and 0 <= c < size and maze[r][c] == 0:
                    return r, c
                for dr, dc in [(1,0),(-1,0),(0,1),(0,-1)]:
                    nr, nc = r+dr, c+dc
                    if (nr, nc) not in seen and 0 <= nr < size and 0 <= nc < size:
                        seen.add((nr, nc))
                        q.append((nr, nc))
            return 1,1

        s_r, s_c = find_nearest_path(1, 1)
        f_r, f_c = find_nearest_path(size - 2, size - 2)

        maze[s_r][s_c] = 2
        maze[f_r][f_c] = 3

        return maze

    def run_camera_analysis(self):
        # self.cap = cv2.VideoCapture(0)
        # # self.camera_analysis_update()
        # if cfg.HAND_DETECTION:
        #     self.HandRecognizer = hand_recognontion.hand_processing()
        # if cfg.EYE_ANALYSIS:
        #     self.EyeAnalyzer = eye_analys.SightDetectionAsync()
        print("Launching Labyrinth camera analysis...")
        functions.run_async(self, self.camera_analysis_async())
    
    async def camera_analysis_async(self):
        # time_start = time.time()
        # print(f"Starting {self.__name__} camera analysis...")
        print(self.width)
        cfg.main_process_loger.info("Starting Labyrithm camera analysis...")
        frame_count = 0
        cfg.CameraProcessor.EYE_ANALYSIS=cfg.EYE_ANALYSIS
        cfg.CameraProcessor.HAND_DETECTION=False
        cfg.CameraProcessor.deque_get_size=1
        # launching = False
        while self.cond_video_process:
            if not cfg.CameraProcessor.last_command_executed:
                x,y = cfg.CameraProcessor.x, cfg.CameraProcessor.y
                if x is not None and y is not None:
                    # print("Target x y:",x,y)
                    self.set_target_position(x,y)
                    cfg.CameraProcessor.last_command_executed = True
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
            await asyncio.sleep(0.1)


# -----------------------
# Запуск
# -----------------------
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # Скрываем главное окно
    # root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}")
    game = LabyrinthGame(root)

    # Пример внешнего управления:
    # def set_external_target(x, y):
    #     game.set_target_position(x, y)

    root.mainloop()