import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from collections import deque
import numpy as np
import threading,  asyncio
import time
import subprocess
import sys, os
import cv2
import screeninfo
# import ping_pong
import ping_pong_iv as ping_pong
import flappy_bird
import TowersGame
import TetrisGame
import DuckHuntGame
import LabyrinthGame
import global_config as cfg
import camera_processing
from pkgs import functions
from pkgs import eye_analys
from pkgs import hand_recognontion
import logging
# import customtkinter as ctk

# === настройки ===
# BUTTON_SIZE = (1000, 980)  # размер каждой "кнопки"
screen_width = screeninfo.get_monitors()[0].width
screen_height = screeninfo.get_monitors()[0].height
BUTTON_SIZE = (int(screen_width/3.2), int(screen_height/2.2))  # размер каждой "кнопки"
IMAGE_PATHS = [
    "app_imgs/flappy_bird.jpg",
    "app_imgs/labyrithm.png",
    "app_imgs/ping-pong.jpg",
    "app_imgs/tetris.png",
    "app_imgs/towers.png",
    "app_imgs/duck_hunt.jpg",
]

# === окно выбора игр ===
MARGIN = 15  # Расстояние между рамками

class GameMenu(tk.Tk):
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger("GameMenu_logger")
        self.title("Выбор игры")
        self.attributes("-fullscreen", True)  # Полноэкранный режим
        self.configure(bg="#3624AD")

        self.canvas = tk.Canvas(self, bg="#216B2A", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self.images = []
        self.buttons = []
        self.hover_index = None
        self.cursor_id = None
        self.cursor_size = 50
        self.cursor_color = "red"

        self.bg_image = None  # фоновое изображение (по желанию)
        self.set_background("app_imgs/background1.jpg")  # включи эту строку, если хочешь фон

        # self.previous_hover_coordinates=None
        # self.eye_hover_coordinates = deque(maxlen=10)

        self.create_buttons()
        self.bind("<Motion>", self.on_mouse_move)
        self.bind("<Escape>", lambda e: self.destroy())
        self.bind("k", lambda e: os.exit(0))
        self.bind("t", self.close_all_games)
        self.run_camera_analysis()
        # if cfg.EYE_ANALYSIS:
        #     self.detecting_eye_direction()


    def close_all_games(self, event=None):
        """Закрывает все открытые игровые окна."""
        for widget in self.winfo_children():
            if isinstance(widget, tk.Toplevel):
                try:
                    widget.close_and_restore()  # используем метод закрытия если есть
                except AttributeError:
                    widget.destroy()  # иначе просто уничтожаем
        cfg.initial_params_to_class()
        cfg.pass_launcher_menu = False
        self.logger.info("All game windows closed by 't' hotkey")
        cfg.main_process_loger.info("All game windows closed by 't' hotkey")

    def set_background(self, image_path: str):
        bg = Image.open(image_path)
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        bg = bg.resize((screen_w, screen_h))
        self.bg_image = ImageTk.PhotoImage(bg)
        self.canvas.create_image(0, 0, image=self.bg_image, anchor="nw")

    def create_buttons(self):
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()

        cols, rows = 3, 2
        total_w = cols * BUTTON_SIZE[0] + (cols - 1) * MARGIN
        total_h = rows * BUTTON_SIZE[1] + (rows - 1) * MARGIN

        start_x = (screen_w - total_w) // 2
        start_y = (screen_h - total_h) // 2

        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= len(IMAGE_PATHS):
                    break

                x = start_x + c * (BUTTON_SIZE[0] + MARGIN)
                y = start_y + r * (BUTTON_SIZE[1] + MARGIN)

                img = Image.open(IMAGE_PATHS[idx]).resize(BUTTON_SIZE)
                img_tk = ImageTk.PhotoImage(img)
                self.images.append(img_tk)

                rect = self.canvas.create_rectangle(
                    x, y, x + BUTTON_SIZE[0], y + BUTTON_SIZE[1],
                    outline="", width=5
                )
                # rect = self.create_rounded_rect(self.canvas, 50, 50, 200, 120, r=25, fill="lightblue", outline="")
                img_id = self.canvas.create_image(x, y, anchor="nw", image=img_tk)

                self.buttons.append((rect, img_id, x, y))

                self.canvas.tag_bind(img_id, "<Button-1>", lambda e, i=idx: self.launch_game(i))
                # self.canvas.tag_bind(rect, "<Button-1>", lambda e, idx=idx: self.on_button_click(e, idx))
                # self.canvas.tag_bind(img_id, "<Button-1>", lambda e, idx=idx: self.on_button_click(e, idx))
            
                idx += 1

    def on_button_click(self, event, index):
        self.logger.info(f"Button clicked at index: {index}")
        self.after(10, lambda: self.launch_game(index))

    def on_mouse_move(self, event):
        self.update_hover(event.x, event.y)

    def update_hover(self, x, y):
        for rect_id, _, bx, by in self.buttons:
            self.canvas.itemconfig(rect_id, outline="")

        for i, (rect_id, img_id, bx, by) in enumerate(self.buttons):
            if bx <= x <= bx + BUTTON_SIZE[0] and by <= y <= by + BUTTON_SIZE[1]:
                self.canvas.itemconfig(rect_id, outline="#3f1ea2")
                self.hover_index = i
                # print("Hover index:", self.hover_index)
                self._draw_cursor(x, y)
                return
        self.hover_index = None
        self._draw_cursor(x, y)

    def _draw_cursor(self, x, y):
        """Create or move a small circle as cursor at (x, y)."""
        # print("Drawing cursor at:", x, y)
        self.logger.info(f"Drawing cursor at: {x}, {y}")
        if x is None or y is None:
            if self.cursor_id is not None:
                try:
                    self.canvas.delete(self.cursor_id)
                except Exception:
                    pass
                self.cursor_id = None
            return
        r = self.cursor_size // 2
        if self.cursor_id is None:
            self.cursor_id = self.canvas.create_oval(x - r, y - r, x + r, y + r,
                                                     fill=self.cursor_color, outline="")
        else:
            self.canvas.coords(self.cursor_id, x - r, y - r, x + r, y + r)

    def launch_game(self, index):
        self.logger.info(f"Launching game at index: {index}")
        print(f"Launching game at index: {index}")
        if index == 2: # Ping-Pong
            # subprocess.Popen([sys.executable, "ping-pong.py"])
            ping_pong.PongGame(self)
        elif index == 0:
            flappy_bird.flappy_bird(self)
        elif index == 4:
            TowersGame.TowersGame(self)
        elif index == 3:
            TetrisGame.TetrisGame(self)
        elif index == 5:
            DuckHuntGame.DuckHuntGame(self)
        elif index == 1:
            LabyrinthGame.LabyrinthGame(self)
        else:
            print(f"Игра {index + 1} пока не подключена")

    def create_rounded_rect(self, canvas, x1, y1, x2, y2, r=20, **kwargs):  # Not used
        points = [
            x1+r, y1,
            x2-r, y1,
            x2, y1,
            x2, y1+r,
            x2, y2-r,
            x2, y2,
            x2-r, y2,
            x1+r, y2,
            x1, y2,
            x1, y2-r,
            x1, y1+r,
            x1, y1
        ]
        return canvas.create_polygon(points, smooth=True, **kwargs)
    
    def run_camera_analysis(self):
        # self.cap = cv2.VideoCapture(0)
        # # self.camera_analysis_update()
        # if cfg.HAND_DETECTION:
        #     self.HandRecognizer = hand_recognontion.hand_processing()
        # if cfg.EYE_ANALYSIS:
        #     self.EyeAnalyzer = eye_analys.SightDetectionAsync()
        functions.run_async(self, self.camera_analysis_async())

    
    def camera_analysis_update(self):
        if self.cap is None:
            raise RuntimeError("Camera capture no fond") 
        ret, img = self.cap.read()
        if ret:
            self.last_frame = img

        self.after(1, self.camera_analysis_update) 

    async def camera_analysis_async(self):
        # time_start = time.time()
        frame_count = 0
        cond= True
        # launching = False
        while cond:
            if cfg.pass_launcher_menu:
                # print("Passing launcher menu...")
                # cfg.main_process_loger.info("Passing launcher menu...")
                await asyncio.sleep(0.1)
                continue
            x, y, = cfg.CameraProcessor.x, cfg.CameraProcessor.y
            cfg.CameraProcessor.EYE_ANALYSIS = cfg.EYE_ANALYSIS
            if x is not None and y is not None and not cfg.CameraProcessor.last_command_executed:
                # print("Camera x,y in main menu:", x, y, cfg.CameraProcessor.EYE_ANALYSIS)
                # canvas_x = int((1-x) * self.canvas.winfo_width())
                # canvas_y = int(y * self.canvas.winfo_height())
                # print("Updating hover to:", x, y)
                self.update_hover(x, y)
                if cfg.CameraProcessor.OK_sighn_detected:
                    launching = True
                    print("Launching game...")
                    cfg.main_process_loger.info(f"Lounching game at index: {self.hover_index}")
                    # if self.hover_index is None:
                    #     print("Hover index is None, cannot launch game.")
                    # self.launch_gme(self.hover_index) # Запуск гри пінг-понг
                    cfg.CameraProcessor.OK_sighn_detected = False
                    cfg.CameraProcessor.last_command_executed = True
                    print("Lounching game at index:", self.hover_index)
                    self.launch_game(self.hover_index if self.hover_index is not None else 2) # Запуск гри пінг-понг
                    # cfg.CameraProcessor.OK_SIGHN_DETECTION = False
                    # cfg.pass_launcher_menu = True
                    # launching = False
                    
                cfg.CameraProcessor.last_command_executed = True
            frame_count += 1
            await asyncio.sleep(0.01)


    async def _camera_analysis_async(self, cap):
        print("Camera running...")
        time_start = time.time()
        frame_count = 0
        while True:
            if frame_count%20==0:
                elapsed = time.time() - time_start
                fps = frame_count / elapsed if elapsed > 0 else 0
                cfg.camera_logger.info(f"Camera FPS: {fps:.2f}")
                # print(f"Camera FPS: {fps:.2f}")
            
            ret, img = cap.read()
            if ret:
                self.last_frame = img
                canvas_x, canvas_y = None, None
                if cfg.EYE_ANALYSIS:
                    x,y  = await self.EyeAnalyzer.image_processing(img.copy())
                    new_x, new_y = self.update_eye_coordinates(x, y)
                    if new_x is not None and new_y is not None:
                        canvas_x = int((1-new_x) * self.canvas.winfo_width())
                        canvas_y = int(new_y * self.canvas.winfo_height())
                        # self.update_hover(canvas_x, canvas_y)
                    # cv2.imshow("Eye Analysis", annotated_img)
                    # cv2.waitKey(1)
                if cfg.HAND_DETECTION:
                    annotated_img = await self.HandRecognizer.recognize_hand(img.copy(), img.copy(), rec_type="Victory_together")
                    cfg.coordinate_logger.info(f"Required pointer: {self.HandRecognizer.required_pointer}")
                    x,y = self.HandRecognizer.required_pointer if self.HandRecognizer.required_pointer else (None, None)
                    if x is not None and y is not None:
                        # if self.previous_hover_coordinates is not None:
                        #     prev_x, prev_y = self.previous_hover_coordinates
                        #     # Вычисляем разницу от предыдущей позиции
                        #     delta_x = x - prev_x
                        #     delta_y = y - prev_y
                        #     amplification = 5.0
                        #     def apply_amplification(prev, delta, amp):
                        #         if delta == 0:
                        #             return prev
                        #         sign = 1 if delta > 0 else -1
                        #         # желаемое смещение после усиления
                        #         desired_move = abs(delta) * amp
                        #         # максимально возможное смещение до границы в ту же сторону
                        #         max_move = (1.0 - prev) if sign > 0 else prev
                        #         # используем минимальное из них
                        #         move = min(desired_move, max_move)
                        #         return min(1.0, max(0.0, prev + sign * move))

                        #     # Применяем усиление отдельно по осям
                        #     x = apply_amplification(prev_x, delta_x, amplification)
                        #     y = apply_amplification(prev_y, delta_y, amplification)
                        canvas_x = int((1-x) * self.canvas.winfo_width())
                        canvas_y = int(y * self.canvas.winfo_height())
                        # if 1-x>0.7 and y>0.5:
                        #     canvas_x = int(0.8 * self.canvas.winfo_width())
                        #     canvas_y = int(0.7 * self.canvas.winfo_height())
                        # elif 1-x>0.7 and y<0.5:
                        #     canvas_x = int(0.8 * self.canvas.winfo_width())
                        #     canvas_y = int(0.3 * self.canvas.winfo_height())
                        # elif 1-x>0.3 and 1-x<0.7 and y>0.5:
                        #     canvas_x = int(0.5 * self.canvas.winfo_width())
                        #     canvas_y = int(0.7 * self.canvas.winfo_height())
                        # elif 1-x>0.7 and 1-x<0.7 and y<0.5:
                        #     canvas_x = int(0.4 * self.canvas.winfo_width())
                        #     canvas_y = int(0.3 * self.canvas.winfo_height())
                        # elif 1-x<0.3 and y<0.5:
                        #     canvas_x = int(0.2 * self.canvas.winfo_width())
                        #     canvas_y = int(0.3 * self.canvas.winfo_height())
                        # elif 1-x<0.3 and y>0.5:
                        #     canvas_x = int(0.2 * self.canvas.winfo_width())
                        #     canvas_y = int(0.7 * self.canvas.winfo_height())
                        # self.update_hover(canvas_x, canvas_y)
                        # self.previous_hover_coordinates=(x,y)
                        
                    print("Required pointer:", self.HandRecognizer.required_pointer)
                    cv2.imshow("Hand Recognition", annotated_img)
                    # cv2.waitKey(1)
            if canvas_x and canvas_y:
                print("Updating hover to:", canvas_x, canvas_y)
                cfg.main_process_loger.info(f"Updating hover to: {canvas_x}, {canvas_y}")
                self.update_hover(canvas_x, canvas_y)
            frame_count += 1
            await asyncio.sleep(0.01)


    def detecting_eye_direction_start(self):
        functions.run_async(self, self.detect_eye_direction_async())

    async def detect_eye_direction_async(self):
        pass

    def update_eye_coordinates(self, new_x: float, new_y: float) -> tuple[float, float]:
        """
        Обновляет очередь координат взгляда и возвращает усредненную позицию.
        Обрабатывает None значения и возвращает среднее только по валидным координатам.
        
        Args:
            new_x (float): новая x координата (0..1) или None
            new_y (float): новая y координата (0..1) или None
                
        Returns:
            tuple[float, float]: (средний x, средний y) или (None, None) если все координаты None
        """
        self.eye_hover_coordinates.append([new_x, new_y])
        
        if len(self.eye_hover_coordinates) > 0:
            # Преобразуем deque в numpy массив
            coords_array = np.array(self.eye_hover_coordinates)
            
            # Проверяем, есть ли хоть одна валидная координата
            valid_coords = ~np.isnan(coords_array).any(axis=1)
            valid_points = coords_array[valid_coords]
            
            # Если все точки None/nan или нет валидных точек
            if len(valid_points) == 0:
                return None, None
                
            # Считаем среднее только по валидным точкам
            mean_x = np.mean(valid_points[:, 0])
            mean_y = np.mean(valid_points[:, 1])
            return mean_x, mean_y

if __name__ == "__main__":
    app = GameMenu()
    app.mainloop()
