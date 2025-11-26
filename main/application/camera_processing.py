import cv2
import time
import numpy as np
import asyncio
import screeninfo
from collections import deque
# import global_config as cfg
from pkgs import eye_analys, hand_recognontion, functions


class CameraProcessor:
    def __init__(self, logger,
                 HAND_DETECTION=True, 
                 EYE_ANALYSIS=True, 
                 OK_SIGHN_DETECTION=True, 
                 VICTORY_TOGETHER_SIGHN_DETECTION=True):
        self.logger = logger
        self.HAND_DETECTION = HAND_DETECTION
        self.EYE_ANALYSIS = EYE_ANALYSIS
        self.OK_SIGHN_DETECTION = OK_SIGHN_DETECTION
        self.VICTORY_TOGETHER_SIGHN_DETECTION = VICTORY_TOGETHER_SIGHN_DETECTION

        self.CLIP_ANALYZER = False

        self.previous_hover_coordinates=None
        self.eye_hover_coordinates = deque(maxlen=10)
        self.deque_get_size = 8
        self.x, self.y = None, None
        self.last_command_executed:bool = True
        self.is_cliped:bool = False

        self.screen = screeninfo.get_monitors()[0]
        self.screen_width, self.screen_height = self.screen.width, self.screen.height

        self.OK_sighn_detected: bool = False
        self.Victoru_together_sighn_detected: bool = False
        

    def run_camera_analysis(self):
        print("Starting camera analysis...")
        self.cap = cv2.VideoCapture(0)
        # self.camera_analysis_update()
        if self.HAND_DETECTION:
            self.HandRecognizer = hand_recognontion.hand_processing()
        if self.EYE_ANALYSIS:
            self.EyeAnalyzer = eye_analys.SightDetectionAsync()
        functions.run_async(self, self.camera_analysis_async(self.cap))

    
    def camera_analysis_update(self):
        if self.cap is None:
            raise RuntimeError("Camera capture no fond") 
        ret, img = self.cap.read()
        if ret:
            self.last_frame = img

        self.after(1, self.camera_analysis_update) 

    async def camera_analysis_async(self, cap):
        print("Starting async camera analysis...")
        time_start = time.time()
        frame_count = 0
        cond= True
        while cond:
            if frame_count%20==0:
                elapsed = time.time() - time_start
                fps = frame_count / elapsed if elapsed > 0 else 0
                self.logger.info(f"Camera FPS: {fps:.2f}")
                # print(f"Camera FPS: {fps:.2f}")
            ret, img = cap.read()
            if not ret:
                self.logger.error("Failed to grab frame")
                frame_count += 1
                continue
            
            self.last_command_executed = False
            try:
                x, y, = await self.image_processing_async(img)
                self.x, self.y = x, y
            except Exception as e:
                self.logger.error(f"Error in image processing async:{e}")
            self.logger.debug("Getting x, y from camera processing:", x, y)
            # if x is not None and y is not None:
            #     # canvas_x = int((1-x) * self.canvas.winfo_width())
            #     # canvas_y = int(y * self.canvas.winfo_height())
            #     # self.update_hover(canvas_x, canvas_y)
            #     pass
            frame_count += 1
            cv2.imshow("Camera", img)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                cond = False
            await asyncio.sleep(0.01)


    async def image_processing_async(self, img):
        # time_start = time.time()
            
        self.last_frame = img
        x, y = None, None
        if self.EYE_ANALYSIS:
            geted_x,geted_y  = await self.EyeAnalyzer.image_processing(img.copy())
            x, y = self.update_eye_coordinates(geted_x, geted_y, self.deque_get_size)
            # print(f"Eye coordinates: {x}, {y}")
            self.logger.info(f"Eye coordinates: {x}, {y}")
            self.x, self.y = x, y

        if self.CLIP_ANALYZER:
            print("Using CLIP_ANALYZER for eye state detection...")
            is_cliped = await self.EyeAnalyzer.eyes_open(img.copy())
            self.is_cliped = is_cliped
            
        if self.HAND_DETECTION:
            # print("Detecting of hand...")
            try:
                # img = await self.HandRecognizer.full_hand_processing(img, img, rec_type="Victory_together")
                lm_landmarks = await self.HandRecognizer.get_hand_landmarks(img)
                if self.OK_SIGHN_DETECTION and lm_landmarks:
                    self.OK_sighn_detected = await self.HandRecognizer.is_OK(lm_landmarks)
                    #print("OK sign detected:", self.OK_sighn_detected) if self.OK_sighn_detected else print("")
                if self.VICTORY_TOGETHER_SIGHN_DETECTION and lm_landmarks:
                    self.Victoru_together_sighn_detected = await self.HandRecognizer.is_fingers_together(lm_landmarks, img.shape)
                    self.logger.info(f"Required pointer: {self.HandRecognizer.required_pointer}")
                    hand_x,hand_y = self.HandRecognizer.required_pointer if self.HandRecognizer.required_pointer else (None, None)
                    if hand_x is not None and hand_y is not None:
                        x = int((1 - hand_x) * self.screen_width) 
                        y = int(hand_y * self.screen_height)
                        # self.previous_hover_coordinates=(x,y)
                        self.logger.info(f"Hand coordinates: {x}, {y}")
                img = await self.HandRecognizer.process_hand_results(lm_landmarks, img, img, "all")
            except Exception as e:
                # print("ERROR HERE:", e)
                self.logger.error(f"Error in hand recognition: {e}")
            
                # self.x, self.y = x, y
            # if x is not None and y is not None:
            #     # if self.previous_hover_coordinates is not None:
            #     #     prev_x, prev_y = self.previous_hover_coordinates
            #     #     # Вычисляем разницу от предыдущей позиции
            #     #     delta_x = x - prev_x
            #     #     delta_y = y - prev_y
            #     #     amplification = 5.0
            #     #     def apply_amplification(prev, delta, amp):
            #     #         if delta == 0:
            #     #             return prev
            #     #         sign = 1 if delta > 0 else -1
            #     #         # желаемое смещение после усиления
            #     #         desired_move = abs(delta) * amp
            #     #         # максимально возможное смещение до границы в ту же сторону
            #     #         max_move = (1.0 - prev) if sign > 0 else prev
            #     #         # используем минимальное из них
            #     #         move = min(desired_move, max_move)
            #     #         return min(1.0, max(0.0, prev + sign * move))

            #     #     # Применяем усиление отдельно по осям
            #     #     x = apply_amplification(prev_x, delta_x, amplification)
            #     #     y = apply_amplification(prev_y, delta_y, amplification)
            #     # canvas_x = int((1-x) * self.canvas.winfo_width())
            #     # canvas_y = int(y * self.canvas.winfo_height())
            #     # if 1-x>0.7 and y>0.5:
            #     #     canvas_x = int(0.8 * self.canvas.winfo_width())
            #     #     canvas_y = int(0.7 * self.canvas.winfo_height())
            #     # elif 1-x>0.7 and y<0.5:
            #     #     canvas_x = int(0.8 * self.canvas.winfo_width())
            #     #     canvas_y = int(0.3 * self.canvas.winfo_height())
            #     # elif 1-x>0.3 and 1-x<0.7 and y>0.5:
            #     #     canvas_x = int(0.5 * self.canvas.winfo_width())
            #     #     canvas_y = int(0.7 * self.canvas.winfo_height())
            #     # elif 1-x>0.7 and 1-x<0.7 and y<0.5:
            #     #     canvas_x = int(0.4 * self.canvas.winfo_width())
            #     #     canvas_y = int(0.3 * self.canvas.winfo_height())
            #     # elif 1-x<0.3 and y<0.5:
            #     #     canvas_x = int(0.2 * self.canvas.winfo_width())
            #     #     canvas_y = int(0.3 * self.canvas.winfo_height())
            #     # elif 1-x<0.3 and y>0.5:
            #     #     canvas_x = int(0.2 * self.canvas.winfo_width())
            #     #     canvas_y = int(0.7 * self.canvas.winfo_height())
            #     self.previous_hover_coordinates=(x,y)
        self.x, self.y = x, y
            # self.logger.debug("Required pointer:", self.HandRecognizer.required_pointer)
        return x, y


    def update_eye_coordinates(self, new_x: float, new_y: float, last_n: int = 10) -> tuple[float, float]:
            """
            Обновляет очередь координат взгляда и возвращает усредненную позицию.
            Обрабатывает None значения и возвращает среднее только по валидным координатам.
            """
            # заменяем None на np.nan и приводим к float — чтобы массив был float dtype
            vx = np.nan if new_x is None else float(new_x)
            vy = np.nan if new_y is None else float(new_y)
            self.eye_hover_coordinates.append([vx, vy])
            
            if len(self.eye_hover_coordinates) == 0:
                return None, None

            last_n = min(last_n, len(self.eye_hover_coordinates))
            recent_list = list(self.eye_hover_coordinates)[-last_n:]
            # создаем numpy-массив float (nan допустим)
            coords_array = np.array(recent_list, dtype=float)
            
            # строки, где нет NaN в любой из колонок
            valid_mask = ~np.isnan(coords_array).any(axis=1)
            valid_points = coords_array[valid_mask]
            
            if valid_points.size == 0:
                return None, None

            mean_x = float(np.mean(valid_points[:, 0]))
            mean_y = float(np.mean(valid_points[:, 1]))
            return mean_x, mean_y