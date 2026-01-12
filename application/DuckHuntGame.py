import tkinter as tk
import random
import time
import asyncio
import global_config as cfg
from pkgs import functions


# --- ОБЯЗАТЕЛЬНО: pip install Pillow ---
from PIL import Image, ImageTk

# ==================================
# Глобальные переменные (Placeholder)
# ==================================
BACKGROUND_IMG = "app_imgs/duck_hunt_bg.png"  # Фон (пока не используется)
DUCK_UP_IMG = "app_imgs/duck_up.png"  # Изображение утки (взмах вверх)
DUCK_DOWN_IMG = "app_imgs/duck_down.png"  # Изображение утки (взмах вниз)
DUCK_DEAD_1_IMG = "app_imgs/duck_dead_1.png"  # Изображение утки (подстрелена, 1 кадр)
DUCK_DEAD_2_IMG = "app_imgs/duck_dead_2.png"  # Изображение утки (падает, 2 кадр)
BULLET_COUNT = 4  # Пуль на раунд
ROUNDS_TO_WIN = 5  # Количество раундов для победы
SCORE_PER_DUCK = 500  # Очки за одну утку

# ==================================
# Параметры игры
# ==================================
DUCK_SIZE = 120 # Размер утки (квадрат)
DUCK_SPEED_MIN = 5  # Минимальная скорость движения утки
DUCK_SPEED_MAX = 15  # Максимальная скорость движения утки
ROUND_TIME_LIMIT = 10000  # Время на раунд в мс (10 секунд)
MAX_DUCKS_PER_ROUND = 2  # Уток, появляющихся одновременно
ANIMATION_FLIP_TIME = 0.5  # Время смены кадра полета (в секундах)
DEATH_ANIM_TIME = 0.5  # Время показа кадра dead_1 (в секундах)


class DuckHuntGame(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Duck Hunt")
        self.attributes('-fullscreen', True)
        self.grab_set()
        self.focus_force()
        self.attributes('-topmost', True)
        cfg.pass_launcher_menu=True

        self.width = self.winfo_screenwidth()
        self.height = self.winfo_screenheight()

        # Canvas
        self.canvas = tk.Canvas(self, width=self.width, height=self.height, bg="#0080FF", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # --- Загрузка спрайтов ---
        # Словарь для хранения ссылок на PhotoImage, чтобы их не съел сборщик мусора
        self.sprite_references = {}
        self.load_sprites()

        # --- Состояние игры ---
        self.score = 0
        self.running = False
        self.game_after_id = None
        self.round = 0
        self.shots_remaining = BULLET_COUNT
        # Список словарей для уток:
        # {'id': (canvas_id), 'x': x, 'y': y, 'vx': vx, 'vy': vy, 'status': 0,
        #  'anim_frame': 'up', 'last_anim_time': 0, 'death_time': 0}
        self.ducks = []
        self.ducks_missed = 0
        self.ducks_hit = 0
        self.cursor = None  # ID прицела
        self.round_start_time = 0

        # --- Внешнее управление ---
        self.cursor_x = self.width // 2
        self.cursor_y = self.height // 2
        self.shoot_signal = False

        # --- Графические элементы ---
        self.score_text_id = None
        self.round_text_id = None
        self.ammo_text_id = None

        # Bindings
        self.bind("<Escape>", lambda e: self.close_and_restore())
        self.bind("<Button-1>", self.on_shoot_mouse)  # Выстрел мышью
        self.bind("<Motion>", self.on_mouse_move)  # Движение мыши для отладки

        self.show_start_screen()

    def load_sprites(self):
        """Загружает, изменяет размер и 'отзеркаливает' все спрайты уток."""
        try:
            # --- Загрузка PIL Images ---
            img_up_right_pil = Image.open(DUCK_UP_IMG).resize((DUCK_SIZE, DUCK_SIZE), Image.Resampling.NEAREST)
            img_down_right_pil = Image.open(DUCK_DOWN_IMG).resize((DUCK_SIZE, DUCK_SIZE), Image.Resampling.NEAREST)
            img_dead1_right_pil = Image.open(DUCK_DEAD_1_IMG).resize((DUCK_SIZE, DUCK_SIZE), Image.Resampling.NEAREST)
            img_dead2_right_pil = Image.open(DUCK_DEAD_2_IMG).resize((DUCK_SIZE, DUCK_SIZE), Image.Resampling.NEAREST)

            # --- Создание отзеркаленных PIL Images ---
            img_up_left_pil = img_up_right_pil.transpose(Image.FLIP_LEFT_RIGHT)
            img_down_left_pil = img_down_right_pil.transpose(Image.FLIP_LEFT_RIGHT)
            img_dead1_left_pil = img_dead1_right_pil.transpose(Image.FLIP_LEFT_RIGHT)
            img_dead2_left_pil = img_dead2_right_pil.transpose(Image.FLIP_LEFT_RIGHT)

            # --- Конвертация в ImageTk.PhotoImage и сохранение ссылок ---
            # (Это обязательно, иначе Tkinter "теряет" изображения)
            self.sprite_references = {
                'up_right': ImageTk.PhotoImage(img_up_right_pil),
                'down_right': ImageTk.PhotoImage(img_down_right_pil),
                'dead1_right': ImageTk.PhotoImage(img_dead1_right_pil),
                'dead2_right': ImageTk.PhotoImage(img_dead2_right_pil),
                'up_left': ImageTk.PhotoImage(img_up_left_pil),
                'down_left': ImageTk.PhotoImage(img_down_left_pil),
                'dead1_left': ImageTk.PhotoImage(img_dead1_left_pil),
                'dead2_left': ImageTk.PhotoImage(img_dead2_left_pil),
            }
        except Exception as e:
            print(f"Ошибка загрузки спрайтов: {e}")
            print("Убедитесь, что файлы .png лежат в папке app_imgs/ и Pillow (PIL) установлен.")
            # Создаем "заглушки", если файлы не найдены
            self.sprite_references = self.create_placeholder_sprites()

    def create_placeholder_sprites(self):
        """Создает цветные квадраты, если спрайты не загрузились."""
        print("Создание заглушек...")
        refs = {}
        colors = {'up_right': 'green', 'down_right': 'yellow', 'dead1_right': 'red',
                  'dead2_right': 'black', 'up_left': 'blue', 'down_left': 'cyan',
                  'dead1_left': 'orange', 'dead2_left': 'grey'}
        for name, color in colors.items():
            img = Image.new('RGBA', (DUCK_SIZE, DUCK_SIZE), color)
            refs[name] = ImageTk.PhotoImage(img)
        return refs

    def get_sprite(self, name):
        """Безопасно получает спрайт из 'хранилища'."""
        return self.sprite_references.get(name)

    # -----------------------
    # Screens and Cleanup
    # -----------------------
    def clear_canvas(self):
        self.canvas.delete("all")
        self.ducks = []
        self.cursor = None

    def close_and_restore(self):
        self.destroy()
        self.master.deiconify()                     # Показать снова
        self.master.lift()                          # Поднять на верх
        self.master.attributes('-topmost', True)    # Временно сделать поверх всех
        self.cond_video_process = False
        cfg.initial_params_to_class()
        cfg.pass_launcher_menu = False
        cfg.main_process_loger.info("Dog-hunt game closed, returning to launcher. \n" \
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
        self.config(cursor="")  # Показываем системный курсор для кнопок

        self.canvas.create_text(self.width // 2, self.height // 3, text="Duck Hunt", fill="yellow",
                                font=("Arial", 48, "bold"))
        rules = (
            f"Правила:\n\n"
            f"1. Отстрелите {MAX_DUCKS_PER_ROUND} уток за раунд.\n"
            f"2. У вас {BULLET_COUNT} патронов на раунд.\n"
            f"3. Для победы нужно пройти {ROUNDS_TO_WIN} раундов.\n"
            "4. Управление: Мышь (по умолчанию) или внешнее управление через камеру.\n\n"
            "Нажмите OK, чтобы начать игру."
        )
        self.canvas.create_text(self.width // 2, int(self.height * 0.55), text=rules, fill="white",
                                font=("Arial", 18), width=int(self.width * 0.7))
        ok_btn = tk.Button(self, text="Начать", font=("Arial", 20), command=self.start_game)
        self.canvas.create_window(self.width // 2, int(self.height * 0.75), window=ok_btn)

    def show_score_screen(self, win=False):
        self.stop_game_loop()
        self.clear_canvas()
        self.config(cursor="")  # Показываем системный курсор для кнопок

        title = "ПОБЕДА!" if win else "Игра окончена"
        text = f"{title}\nФинальный счёт: {self.score}\nРаундов пройдено: {self.round}"
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
        self.round = 0
        self.running = True

        self.next_round()

    def next_round(self):
        """Переход к следующему раунду."""
        self.running = True
        if not self.running: return

        self.round += 1
        if self.round > ROUNDS_TO_WIN:
            self.show_score_screen(win=True)
            return

        self.config(cursor="none")  # Прячем системный курсор на время раунда

        self.shots_remaining = BULLET_COUNT
        self.ducks_hit = 0
        self.ducks_missed = 0

        # Удаляем старые утки
        for duck in self.ducks:
            self.canvas.delete(duck['id'])
        self.ducks = []

        self.spawn_duck(self.round)  # Спавним первую утку

        self.draw_info()
        self.draw_cursor()  # <-- Эта функция теперь исправлена

        self.round_start_time = time.time()
        self.schedule_next_frame()

    def end_round(self):
        """Завершает текущий раунд и проверяет условия."""
        self.stop_game_loop()

        # Если не подстрелено достаточно уток -> Game Over
        if self.ducks_hit < MAX_DUCKS_PER_ROUND:
            self.show_score_screen(win=False)
        else:
            # Успешный переход к следующему раунду
            self.after(1000, self.next_round)  # Пауза 2 секунды перед следующим

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
    def schedule_next_frame(self):
        # ~ 60 FPS
        self.game_after_id = self.after(16, self.game_loop)

    def game_loop(self):
        if not self.running:
            return

        current_time = time.time()

        # 1. Проверяем выстрел
        if self.shoot_signal:
            self.process_shoot()
            self.shoot_signal = False

        # 2. Движение уток
        self.update_ducks_movement(current_time)

        # 3. Проверка конца раунда (время или все утки убиты)
        elapsed = (current_time - self.round_start_time)
        if elapsed > (ROUND_TIME_LIMIT / 1000) or self.ducks_hit >= MAX_DUCKS_PER_ROUND:
            self.end_round()
            return

        # 4. Обновление графики
        self.draw_info()
        self.update_cursor()

        # 5. Продолжение цикла
        self.schedule_next_frame()

    def draw_info(self):
        """Отображает счёт, раунд и патроны."""
        if self.score_text_id: self.canvas.delete(self.score_text_id)
        if self.round_text_id: self.canvas.delete(self.round_text_id)
        if self.ammo_text_id: self.canvas.delete(self.ammo_text_id)

        self.score_text_id = self.canvas.create_text(50, 50, text=f"Счёт: {self.score}",
                                                     fill="white", font=("Arial", 20), anchor="nw")
        self.round_text_id = self.canvas.create_text(self.width // 2, 50, text=f"Раунд: {self.round}/{ROUNDS_TO_WIN}",
                                                     fill="white", font=("Arial", 20), anchor="n")
        self.ammo_text_id = self.canvas.create_text(self.width - 50, 50, text=f"Патроны: {self.shots_remaining}",
                                                    fill="white", font=("Arial", 20), anchor="ne")

    def draw_cursor(self):
        """Рисует прицел (крестик)."""
        # (ИСПРАВЛЕНО) Удаляем СТАРЫЙ курсор по тэгу, а не по переменной
        self.canvas.delete("cursor")

        R = 15  # Размер крестика
        # Создаем НОВЫЙ курсор и сохраняем его ID
        self.cursor = (
            self.canvas.create_line(
                self.cursor_x - R, self.cursor_y, self.cursor_x + R, self.cursor_y,
                fill="red", width=2, tags="cursor"
            ),
            self.canvas.create_line(
                self.cursor_x, self.cursor_y - R, self.cursor_x, self.cursor_y + R,
                fill="red", width=2, tags="cursor"
            )
        )

    def update_cursor(self):
        """Перемещает прицел, используя текущие координаты self.cursor_x/y."""
        if self.cursor:
            R = 15
            # Перемещаем обе линии крестика
            self.canvas.coords(self.cursor[0],
                               self.cursor_x - R, self.cursor_y, self.cursor_x + R, self.cursor_y)
            self.canvas.coords(self.cursor[1],
                               self.cursor_x, self.cursor_y - R, self.cursor_x, self.cursor_y + R)

    # -----------------------
    # Duck Logic
    # -----------------------
    def spawn_duck(self, difficulty_multiplier):
        """Создает новую утку."""

        start_x = -DUCK_SIZE  # За левой границей
        start_y = random.randint(self.height // 3, self.height - DUCK_SIZE - 50)

        # Случайное движение (диагональ)
        vx = random.randint(DUCK_SPEED_MIN, DUCK_SPEED_MAX) + difficulty_multiplier
        vy = random.choice([-1, 1]) * random.randint(DUCK_SPEED_MIN // 2, DUCK_SPEED_MAX // 2)

        # Рисуем утку, используя спрайт
        initial_sprite = self.get_sprite('up_right' if vx > 0 else 'up_left')
        duck_id = self.canvas.create_image(
            start_x, start_y,
            image=initial_sprite,
            anchor="nw"  # Важно для совпадения координат
        )

        # Статус: 0 - жива, 1 - подстрелена (кадр 1), 2 - падает (кадр 2), 3 - улетела
        self.ducks.append({
            'id': duck_id,
            'x': start_x,
            'y': start_y,
            'vx': vx,
            'vy': vy,
            'status': 0,
            'anim_frame': 'up',
            'last_anim_time': time.time(),
            'death_time': 0
        })

    def update_ducks_movement(self, current_time):
        """Двигает уток и обрабатывает их поведение."""
        ducks_to_keep = []
        something_removed = False

        for duck in self.ducks:
            duck_id = duck['id']

            if duck['status'] == 0:  # Если жива
                new_x = duck['x'] + duck['vx']
                new_y = duck['y'] + duck['vy']

                # Отскок от верхней/нижней границы
                if new_y <= 0 or new_y >= self.height - DUCK_SIZE:
                    duck['vy'] = -duck['vy']
                    new_y = duck['y'] + duck['vy']  # Пересчитываем new_y

                # Улетела за правую (или левую) границу -> промах
                if new_x > self.width or new_x < -DUCK_SIZE:
                    self.canvas.delete(duck_id)
                    self.ducks_missed += 1
                    something_removed = True
                    if self.ducks_missed + self.ducks_hit < MAX_DUCKS_PER_ROUND:
                        self.spawn_duck(self.round)
                else:
                    # Анимация и разворот
                    direction = 'right' if duck['vx'] > 0 else 'left'

                    # Смена кадра анимации
                    if current_time - duck['last_anim_time'] > ANIMATION_FLIP_TIME:
                        duck['anim_frame'] = 'down' if duck['anim_frame'] == 'up' else 'up'
                        duck['last_anim_time'] = current_time

                    sprite_name = f"{duck['anim_frame']}_{direction}"
                    self.canvas.itemconfig(duck_id, image=self.get_sprite(sprite_name))

                    # Движение
                    self.canvas.move(duck_id, duck['vx'], duck['vy'])
                    duck['x'] = new_x
                    duck['y'] = new_y
                    ducks_to_keep.append(duck)

            elif duck['status'] == 1:  # Подстрелена (кадр 1)
                # Ждем DEATH_ANIM_TIME секунд
                if current_time - duck['death_time'] > DEATH_ANIM_TIME:
                    duck['status'] = 2  # Переключаем на "падение"

                    # Определяем направление, в котором утка была подстрелена
                    direction = 'right' if duck['vx'] > 0 else 'left'
                    self.canvas.itemconfig(duck_id, image=self.get_sprite(f'dead2_{direction}'))

                ducks_to_keep.append(duck)

            elif duck['status'] == 2:  # Падает (кадр 2)
                fall_speed = 10
                new_y = duck['y'] + fall_speed

                if new_y > self.height:
                    # Упала за экран
                    self.canvas.delete(duck_id)
                    something_removed = True
                else:
                    self.canvas.move(duck_id, 0, fall_speed)
                    duck['y'] = new_y
                    ducks_to_keep.append(duck)

        # Обновляем список уток
        self.ducks = ducks_to_keep

    def check_hit(self):
        """Проверяет, попал ли выстрел по утке."""
        hit_duck_index = -1

        for i, duck in enumerate(self.ducks):
            if duck['status'] == 0:  # Проверяем только живых
                x, y = duck['x'], duck['y']
                # Простая AABB (квадратная) проверка попадания
                if x <= self.cursor_x <= x + DUCK_SIZE and y <= self.cursor_y <= y + DUCK_SIZE:
                    hit_duck_index = i
                    break  # Попали, выходим

        return hit_duck_index

    def process_shoot(self):
        """Логика выстрела."""
        if self.shots_remaining <= 0:
            return  # Нет патронов

        self.shots_remaining -= 1

        hit_index = self.check_hit()

        if hit_index != -1:
            # Утка убита
            duck = self.ducks[hit_index]
            duck_id = duck['id']

            # Меняем статус на "подстрелена" (кадр 1)
            duck['status'] = 1
            duck['death_time'] = time.time()

            # Определяем направление для спрайта dead_1
            direction = 'right' if duck['vx'] > 0 else 'left'
            self.canvas.itemconfig(duck_id, image=self.get_sprite(f'dead1_{direction}'))

            # Останавливаем движение (vx сохраняем для определения спрайта падения)
            duck['vy'] = 0

            self.score += SCORE_PER_DUCK
            self.ducks_hit += 1

            # Если это не последняя необходимая утка, спавним следующую
            if self.ducks_hit < MAX_DUCKS_PER_ROUND:
                # Спавним не сразу, а с небольшой задержкой
                self.after(500, lambda: self.spawn_duck(self.round))

                # -----------------------

    # Input Handlers
    # -----------------------
    def on_mouse_move(self, event):
        """Для отладки и управления по умолчанию."""
        self.set_cursor_position(event.x, event.y)

    def on_shoot_mouse(self, event):
        """Для отладки и управления по умолчанию."""
        self.set_control_action('shoot')

    # -----------------------
    # External API (Для управления через камеру)
    # -----------------------
    def set_cursor_position(self, x: int | float, y: int | float):
        """
        Устанавливает позицию прицела в экранных пикселях.
        """
        # Clamp (ограничение)
        self.cursor_x = max(0, min(self.width, int(x)))
        self.cursor_y = max(0, min(self.height, int(y)))

    def set_control_action(self, action: str):
        """
        Метод для внешнего управления.

        Параметры:
            action: 'shoot' для выстрела.
        """
        if action == 'shoot' and self.running and self.shots_remaining > 0:
            # Используем after_idle для безопасного вызова в главном потоке
            self.after_idle(lambda: setattr(self, 'shoot_signal', True))

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
        cfg.main_process_loger.info("Starting Dog-Hunt camera analysis...")
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
    # Скрываем главное окно, так как игра будет в Toplevel
    root.withdraw()

    game = DuckHuntGame(root)
    root.mainloop()