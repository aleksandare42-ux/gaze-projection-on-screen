import tkinter as tk
import random
import time
import asyncio
import global_config as cfg
from pkgs import functions

# from PIL import Image, ImageTk

# ==================================
# Глобальные переменные (Placeholder)
# ==================================
GAME_AREA_WIDTH = 10  # Ширина игрового поля в блоках
GAME_AREA_HEIGHT = 20  # Высота игрового поля в блоках
BLOCK_SIZE = 70  # Размер одного блока в пикселях
INITIAL_FALL_SPEED = 500  # Время падения (в миллисекундах)
SCORE_PER_LINE = 100

# Цвета для каждой фигуры (и пустой ячейки)
COLORS = {
    0: "gray",  # Пустая ячейка
    1: "cyan",  # I
    2: "blue",  # J
    3: "orange",  # L
    4: "yellow",  # O
    5: "green",  # S
    6: "purple",  # T
    7: "red"  # Z
}

# Формы тетримино (Y, X)
SHAPES = [
    # I
    [[0, 0, 0, 0],
     [1, 1, 1, 1],
     [0, 0, 0, 0],
     [0, 0, 0, 0]],
    # J
    [[2, 0, 0],
     [2, 2, 2],
     [0, 0, 0]],
    # L
    [[0, 0, 3],
     [3, 3, 3],
     [0, 0, 0]],
    # O
    [[4, 4],
     [4, 4]],
    # S
    [[0, 5, 5],
     [5, 5, 0],
     [0, 0, 0]],
    # T
    [[0, 6, 0],
     [6, 6, 6],
     [0, 0, 0]],
    # Z
    [[7, 7, 0],
     [0, 7, 7],
     [0, 0, 0]]
]


class TetrisGame(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Tetris Game")
        self.attributes('-fullscreen', True)
        self.grab_set()
        self.focus_force()
        self.attributes('-topmost', True)
        cfg.pass_launcher_menu=True

        self.screen_width = self.winfo_screenwidth()
        self.screen_height = self.winfo_screenheight()

        # Размеры игровой области
        self.canvas_width = GAME_AREA_WIDTH * BLOCK_SIZE
        self.canvas_height = GAME_AREA_HEIGHT * BLOCK_SIZE
        self.x_offset = (self.screen_width - self.canvas_width) // 2
        self.y_offset = (self.screen_height - self.canvas_height) // 2

        # Canvas
        self.canvas = tk.Canvas(self, width=self.screen_width, height=self.screen_height, bg="black",
                                highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # --- Состояние игры ---
        self.board = []  # Основное поле (список списков с кодами цветов/фигур)
        self.current_piece = None  # Текущая фигура (матрица)
        self.current_piece_x = 0  # Координата X верхнего левого угла фигуры на поле
        self.current_piece_y = 0  # Координата Y верхнего левого угла фигуры на поле
        self.score = 0
        self.running = False
        self.game_after_id = None

        # --- Внешнее управление ---
        self.control_actions = []  # Список действий для выполнения на следующем кадре

        # --- Графические элементы ---
        self.block_refs = {}  # Словарь для хранения ID нарисованных блоков
        self.score_text_id = None

        # Bindings (сразу на методы-обработчики)
        self.bind("<Left>", lambda e: self.set_control_action('move_left'))
        self.bind("<Right>", lambda e: self.set_control_action('move_right'))
        self.bind("<Up>", lambda e: self.set_control_action('rotate'))
        self.bind("<Down>", lambda e: self.set_control_action('drop'))
        self.bind("<Escape>", lambda e: self.close_and_restore())

        self.show_start_screen()

    # -----------------------
    # Screens and Cleanup
    # -----------------------
    def clear_canvas(self):
        """Очищает холст и сбрасывает все объекты."""
        self.canvas.delete("all")
        self.block_refs = {}
        self.score_text_id = None

    def close_and_restore(self):
        self.destroy()
        self.master.deiconify()                     # Показать снова
        self.master.lift()                          # Поднять на верх
        self.master.attributes('-topmost', True)    # Временно сделать поверх всех
        self.cond_video_process = False
        cfg.initial_params_to_class()
        cfg.pass_launcher_menu = False
        cfg.main_process_loger.info("Tetris game closed, returning to launcher. \n" \
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
        self.canvas.create_text(self.screen_width // 2, self.screen_height // 3, text="Тетрис", fill="white",
                                font=("Arial", 48, "bold"))
        rules = (
            "Правила:\n\n"
            "1. Управление: стрелки Влево/Вправо (движение), Вверх (поворот), Вниз (сброс).\n"
            "2. Вы можете управлять через внешний API, вызывая set_control_action().\n"
            "3. Игра заканчивается, когда фигуры достигают верха поля.\n\n"
            "Нажмите OK, чтобы начать игру."
        )
        self.canvas.create_text(self.screen_width // 2, int(self.screen_height * 0.55), text=rules, fill="white",
                                font=("Arial", 18), width=int(self.screen_width * 0.7))
        ok_btn = tk.Button(self, text="Начать", font=("Arial", 20), command=self.start_game)
        self.canvas.create_window(self.screen_width // 2, int(self.screen_height * 0.75), window=ok_btn)

    def show_score_screen(self):
        self.stop_game_loop()
        self.clear_canvas()
        text = f"Игра окончена. Счёт: {self.score}"
        self.canvas.create_text(self.screen_width // 2, self.screen_height // 3, text=text, fill="white",
                                font=("Arial", 40))
        restart_btn = tk.Button(self, text="Restart", font=("Arial", 18), command=self.start_game)
        exit_btn = tk.Button(self, text="Exit", font=("Arial", 18), command=self.close_and_restore)
        self.canvas.create_window(self.screen_width // 2 - 100, int(self.screen_height * 0.6), window=restart_btn)
        self.canvas.create_window(self.screen_width // 2 + 100, int(self.screen_height * 0.6), window=exit_btn)

    # -----------------------
    # Game lifecycle
    # -----------------------
    def start_game(self):
        self.clear_canvas()
        self.score = 0
        self.running = True

        # Инициализация поля нулями (пустые ячейки)
        self.board = [[0] * GAME_AREA_WIDTH for _ in range(GAME_AREA_HEIGHT)]

        self.draw_game_area()
        self.update_score_display()

        self.new_piece()
        self.game_loop_auto_fall()

    def stop_game_loop(self):
        self.running = False
        if self.game_after_id is not None:
            try:
                self.after_cancel(self.game_after_id)
            except Exception:
                pass
            self.game_after_id = None

    # -----------------------
    # Core Tetris Logic
    # -----------------------

    def new_piece(self):
        """Создает новую случайную фигуру."""
        self.current_piece = random.choice(SHAPES)
        # Начинаем сверху по центру
        self.current_piece_x = GAME_AREA_WIDTH // 2 - len(self.current_piece[0]) // 2
        self.current_piece_y = 0

        if not self.check_collision(0, 0, self.current_piece):
            self.draw_piece()
        else:
            # Game Over, если новая фигура не может появиться
            self.show_score_screen()

    def check_collision(self, dx, dy, piece):
        """Проверяет, возможен ли сдвиг фигуры на (dx, dy) или ее установка."""
        for y in range(len(piece)):
            for x in range(len(piece[0])):
                if piece[y][x] != 0:  # Если это блок фигуры
                    new_x = self.current_piece_x + x + dx
                    new_y = self.current_piece_y + y + dy

                    # Проверка границ поля
                    if new_x < 0 or new_x >= GAME_AREA_WIDTH or new_y >= GAME_AREA_HEIGHT:
                        return True

                    # Проверка занятости ячейки (игнорируем ячейки, которые выше поля)
                    if new_y >= 0 and self.board[new_y][new_x] != 0:
                        return True
        return False

    def rotate_piece(self, piece):
        """Поворачивает матрицу фигуры на 90 градусов по часовой стрелке."""
        N = len(piece)
        new_piece = [[0] * N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                new_piece[j][N - 1 - i] = piece[i][j]
        return new_piece

    def lock_piece(self):
        """Блокирует текущую фигуру на поле и проверяет собранные линии."""
        for y in range(len(self.current_piece)):
            for x in range(len(self.current_piece[0])):
                if self.current_piece[y][x] != 0:
                    board_x = self.current_piece_x + x
                    board_y = self.current_piece_y + y

                    # Проверяем, чтобы не выйти за верхнюю границу, хотя это уже Game Over
                    if 0 <= board_y < GAME_AREA_HEIGHT:
                        self.board[board_y][board_x] = self.current_piece[y][x]

        self.clear_lines()
        self.new_piece()

    def clear_lines(self):
        """Проверяет и удаляет собранные линии."""
        lines_cleared = 0
        new_board = []

        # Идем снизу вверх
        for row in reversed(self.board):
            if 0 not in row:
                # Линия полная -> удаляем
                lines_cleared += 1
            else:
                # Линия не полная -> оставляем
                new_board.insert(0, row)

        if lines_cleared > 0:
            # Добавляем пустые линии сверху
            for _ in range(lines_cleared):
                new_board.insert(0, [0] * GAME_AREA_WIDTH)

            self.board = new_board
            self.score += SCORE_PER_LINE * lines_cleared
            self.redraw_board()  # Перерисовываем все поле
            self.update_score_display()

    # -----------------------
    # Game Loop and Input Handling
    # -----------------------

    def game_loop_auto_fall(self):
        """Основной цикл падения фигуры."""
        if not self.running:
            return

        # 1. Обрабатываем внешние действия (горизонталь, поворот, сброс)
        self.process_control_actions()

        # 2. Фигура падает вниз на 1
        if not self.move_piece(0, 1):
            # Если падение невозможно, фигура блокируется
            self.lock_piece()

        # 3. Планируем следующий кадр падения
        self.game_after_id = self.after(INITIAL_FALL_SPEED, self.game_loop_auto_fall)

    def process_control_actions(self):
        """Выполняет все запланированные действия (от внешнего API или кнопок)."""
        # Действия обрабатываются в порядке, в котором они были добавлены
        for action in self.control_actions:
            if action == 'move_left':
                self.move_piece(-1, 0)
            elif action == 'move_right':
                self.move_piece(1, 0)
            elif action == 'rotate':
                self.attempt_rotate()
            elif action == 'drop':
                self.hard_drop()
        self.control_actions = []  # Очищаем список после выполнения

    def move_piece(self, dx, dy):
        """Пытается сдвинуть фигуру, возвращает True, если успешно."""
        if not self.check_collision(dx, dy, self.current_piece):
            self.undraw_piece()
            self.current_piece_x += dx
            self.current_piece_y += dy
            self.draw_piece()
            return True
        return False

    def attempt_rotate(self):
        """Пытается повернуть фигуру, используя простой 'кик'."""
        if self.current_piece is None: return

        new_piece = self.rotate_piece(self.current_piece)

        # Проверяем сдвиги (kick/wall kick)
        offsets = [(0, 0), (-1, 0), (1, 0), (0, -1)]  # (dx, dy)

        for dx, dy in offsets:
            if not self.check_collision(dx, dy, new_piece):
                self.undraw_piece()
                self.current_piece_x += dx
                self.current_piece_y += dy
                self.current_piece = new_piece
                self.draw_piece()
                return

    def hard_drop(self):
        """Мгновенно сбрасывает фигуру вниз."""
        if self.current_piece is None: return

        while self.move_piece(0, 1):
            pass
        # После сброса фигура сразу блокируется
        self.lock_piece()



    # -----------------------
    # Drawing (Graphics)
    # -----------------------
    def draw_game_area(self):
        """Рисует границы игрового поля."""
        x1 = self.x_offset
        y1 = self.y_offset
        x2 = self.x_offset + self.canvas_width
        y2 = self.y_offset + self.canvas_height

        # Ободок
        self.canvas.create_rectangle(x1, y1, x2, y2, outline="white", width=2)

    def update_score_display(self):
        """Обновляет отображение счёта."""
        if self.score_text_id:
            self.canvas.delete(self.score_text_id)

        text = f"Счёт: {self.score}"
        self.score_text_id = self.canvas.create_text(
            self.x_offset + self.canvas_width + 50, self.y_offset + 50,
            text=text, fill="white",
            font=("Arial", 20), anchor="nw"
        )

    def draw_block(self, board_x, board_y, color_index):
        """Рисует один блок на холсте и сохраняет его ID."""
        x1 = self.x_offset + board_x * BLOCK_SIZE
        y1 = self.y_offset + board_y * BLOCK_SIZE
        x2 = x1 + BLOCK_SIZE
        y2 = y1 + BLOCK_SIZE

        block_id = self.canvas.create_rectangle(x1, y1, x2, y2,
                                                fill=COLORS.get(color_index, 'gray'),
                                                outline="black")
        self.block_refs[(board_x, board_y)] = block_id
        return block_id

    def undraw_block(self, board_x, board_y):
        """Удаляет блок с холста."""
        if (board_x, board_y) in self.block_refs:
            self.canvas.delete(self.block_refs[(board_x, board_y)])
            del self.block_refs[(board_x, board_y)]

    def draw_piece(self):
        """Рисует текущую движущуюся фигуру."""
        if self.current_piece is None: return

        for y in range(len(self.current_piece)):
            for x in range(len(self.current_piece[0])):
                color_index = self.current_piece[y][x]
                if color_index != 0:
                    board_x = self.current_piece_x + x
                    board_y = self.current_piece_y + y
                    if 0 <= board_y < GAME_AREA_HEIGHT and 0 <= board_x < GAME_AREA_WIDTH:
                        self.draw_block(board_x, board_y, color_index)

    def undraw_piece(self):
        """Удаляет текущую движущуюся фигуру (для перерисовки)."""
        if self.current_piece is None: return

        for y in range(len(self.current_piece)):
            for x in range(len(self.current_piece[0])):
                if self.current_piece[y][x] != 0:
                    board_x = self.current_piece_x + x
                    board_y = self.current_piece_y + y
                    self.undraw_block(board_x, board_y)

    def redraw_board(self):
        """Перерисовывает все заблокированные блоки на поле (после очистки линий)."""
        # Сначала очищаем все заблокированные блоки
        keys_to_delete = [k for k, v in self.block_refs.items() if self.canvas.type(v) == 'rectangle']
        for k in keys_to_delete:
            self.undraw_block(k[0], k[1])

        # Затем рисуем новые
        for y in range(GAME_AREA_HEIGHT):
            for x in range(GAME_AREA_WIDTH):
                color_index = self.board[y][x]
                if color_index != 0:
                    self.draw_block(x, y, color_index)

    # -----------------------
    # External API
    # -----------------------
    def set_control_action(self, action: str):
        """
        Метод для внешнего управления.

        Параметры:
            action: 'move_left', 'move_right', 'rotate', 'drop'.
        """
        valid_actions = ['move_left', 'move_right', 'rotate', 'drop']
        if action in valid_actions and self.running:
            # Используем after_idle для безопасного добавления действия
            # в список, который будет обработан в следующем game_loop
            self.after_idle(lambda: self.control_actions.append(action))

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
    # Устанавливаем размер окна root, чтобы он не был минимальным
    root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}")
    game = TetrisGame(root)

    # Пример внешнего управления:
    # root.after(5000, lambda: game.set_control_action('rotate'))

    root.mainloop()