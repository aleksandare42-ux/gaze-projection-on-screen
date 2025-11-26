import cv2
import sys
# sys.stderr = open(os.devnull, 'w')
import mediapipe as mp
# from pckgs import hand_signs
import math
import logging
import os


class hand_processing:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils  # для отрисовки точек на изображении
        self.last_not_none_pointer = None, None
        self.required_pointer = None
        self.logger = logging.getLogger("Hand_Recognition_logger")


    async def full_hand_processing(self, img, drawing_img=None, rec_type="all"):
        self.logger.debug(f"Hand recognition processing frame, rec_type={rec_type}")
        lm_list = await self.get_hand_landmarks(img)
        drawing_img = await self.process_hand_results(lm_list,img, drawing_img, rec_type)
        return drawing_img


    async def get_hand_landmarks(self, img):
        if len(img.shape)>2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # конвертация в RGB для MediaPipe
        else:
            img_rgb = img
        results = self.hands.process(img_rgb)  # передаём кадр в распознаватель рук
        # self.required_pointer = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                # Получаем координаты всех точек
                lm_list = []
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, _ = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append((id, cx, cy))
            return lm_list

    async def process_hand_results(self, lm_list, img, drawing_img=None, *args):
        if lm_list:
            if "fist_rec" in args:
                if await self.is_fist(lm_list):
                    cv2.putText(drawing_img, 'Fist Detected!', (20, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
            if "OK_rec" in args:
                if await self.is_OK(lm_list):
                    cv2.putText(drawing_img, 'OK Sign Detected!', (20, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    # print("OK sign detected")
            if "Victory_together" in args:
                if await self.is_fingers_together(lm_list, img.shape):
                    cv2.putText(drawing_img, 'Victory Sign Detected!', (20, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            if "all" in args:
                if await self.is_fist(lm_list):
                    cv2.putText(drawing_img, 'Fist Detected!', (20, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
                elif await self.is_OK(lm_list):
                    cv2.putText(drawing_img, 'OK Sign Detected!', (20, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                elif await self.is_fingers_together(lm_list, img.shape):
                    cv2.putText(drawing_img, 'Victory-together sign Detected!', (20, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    
        return drawing_img
                            

    async def is_fist(self,lm_list):
        """
        Проверка кулака независимо от ориентации камеры
        """
        if not lm_list:
            return False

        wrist = lm_list[0][1], lm_list[0][2]

        # id кончиков пальцев (кроме большого)
        tip_ids = [8, 12, 16, 20]
        pip_ids = [6, 10, 14, 18]

        fingers_folded = 0

        for tip, pip in zip(tip_ids, pip_ids):
            tip_pos = lm_list[tip][1], lm_list[tip][2]
            pip_pos = lm_list[pip][1], lm_list[pip][2]

            # Вычисляем расстояние от запястья до кончика и до PIP-сустава
            dist_tip = math.hypot(tip_pos[0] - wrist[0], tip_pos[1] - wrist[1])
            dist_pip = math.hypot(pip_pos[0] - wrist[0], pip_pos[1] - wrist[1])

            # Если кончик ближе к запястью, чем PIP — палец согнут
            if dist_tip < dist_pip:
                fingers_folded += 1

        # Если все 4 пальца согнуты — кулак
        return fingers_folded == 4

    async def is_OK(self, lm_list, rat=0.35):
        """
        Проверка на знак ОК
        """
        thumb_tip = lm_list[4][1], lm_list[4][2]
        index_tip = lm_list[8][1], lm_list[8][2]

        wrist = lm_list[0][1], lm_list[0][2]
        middle_mcp = lm_list[9][1], lm_list[9][2]

        tip_distance = math.hypot(index_tip[0] - thumb_tip[0], index_tip[1] - thumb_tip[1])
        hand_size = math.hypot(middle_mcp[0] - wrist[0], middle_mcp[1] - wrist[1])

        ratio = tip_distance / hand_size
        
        if ratio < rat:
            return True
        else:
            return False
    
    async def is_fingers_together(self, lm_list, img_shape=None,  max_dist=40, min_height=50):
        """
        Проверка жеста: указательный и средний пальцы подняты и соединены вместе
        
        Args:
            lm_list: список координат ключевых точек руки
            max_dist: максимальное расстояние между кончиками пальцев (в пикселях)
            min_height: минимальная высота поднятых пальцев относительно основания (в пикселях)
        """
        # Получаем координаты кончиков указательного и среднего пальцев
        index_tip = lm_list[8][1], lm_list[8][2]  # указательный
        middle_tip = lm_list[12][1], lm_list[12][2]  # средний
        
        # Координаты оснований пальцев (для проверки высоты)
        index_base = lm_list[5][1], lm_list[5][2]
        middle_base = lm_list[9][1], lm_list[9][2]
        
        # Расстояние между кончиками пальцев
        tips_distance = math.hypot(index_tip[0] - middle_tip[0], 
                                index_tip[1] - middle_tip[1])
        
        # Высота каждого пальца (расстояние от основания до кончика)
        index_height = math.hypot(index_tip[0] - index_base[0],
                                index_tip[1] - index_base[1])
        middle_height = math.hypot(middle_tip[0] - middle_base[0],
                                middle_tip[1] - middle_base[1])
        
        # Проверяем условия:
        # 1. Пальцы достаточно близко друг к другу
        # 2. Оба пальца подняты на достаточную высоту
        if (tips_distance < max_dist and 
            index_height > min_height and 
            middle_height > min_height):
            self._update_required_pointer(lm_list, img_shape)
            return True
        
        self.required_pointer=None
        return False
    
    def _update_required_pointer(self, lm_list, img_shape=None):
        """
        Вычисляет и сохраняет в self.required_pointer середину между кончиками
        указательного и среднего пальцев.
        Если передан img_shape (h,w,...) — сохраняет нормализованные координаты (x_norm,y_norm) в диапазоне [0,1].
        Иначе сохраняет пиксельные координаты.
        """
        if not lm_list or len(lm_list) <= 12:
            self.required_pointer = None
            return
        ix, iy = lm_list[8][1], lm_list[8][2]
        mx, my = lm_list[12][1], lm_list[12][2]
        # средняя точка (int пиксели)
        cx = (ix + mx) / 2.0
        cy = (iy + my) / 2.0

        if img_shape is not None:
            h, w = img_shape[0], img_shape[1]
            # нормализация и клэмпинг в [0,1]
            nx = min(max(cx / w, 0.0), 1.0)
            ny = min(max(cy / h, 0.0), 1.0)
            self.required_pointer = (nx, ny)
            self.last_not_none_pointer = (nx, ny)
        else:
            self.last_not_none_pointer = (cx, cy)
            self.required_pointer = (int(cx), int(cy))


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = hand_processing()

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    connections = mp_hands.HAND_CONNECTIONS

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # зеркалим для удобства
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.hands.process(frame_rgb)

        ok_detected = False

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Отрисовка всех точек и связей
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    connections,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )

                # Собираем точки для проверки жеста "ОК"
                lm_list = []
                h, w, _ = frame.shape
                for id, lm in enumerate(hand_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append((id, cx, cy))

                if detector.is_OK(lm_list):
                    ok_detected = True

        if ok_detected:
            cv2.putText(frame, "OK Sign Detected!", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        cv2.imshow("Hand Demo", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()