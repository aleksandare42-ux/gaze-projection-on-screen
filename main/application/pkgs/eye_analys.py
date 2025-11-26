import cv2
import sys, os, time, logging, asyncio
import random, screeninfo
# sys.stderr = open(os.devnull, 'w')
import mediapipe as mp
from datetime import datetime

import torch
from torchvision import transforms as T
import numpy as np
from ultralytics import YOLO
import face_alignment
from pkgs.sighn_analys import AFFGazeNet as AFF1
from pkgs.sighn_analys import estimate_head_pose as est1
from pkgs.sighn_analys import crop_eye

def wrapper(func, *args,**kwargs):
    def inner(*args, **kwargs):
        print(f"Starting function {func.__name__}...")
        start = time.time()
        res = func(*args, **kwargs)
        finish = time.time()
        print(f"Function {func.__name__} takes: {finish - start:.3f} seconds to execute")
        return res
    return inner

class ClipAnalyzer:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        # --- точки вокруг глаз (по индексу mediapipe) ---
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [263, 387, 385, 362, 380, 373]

    def eye_aspect_ratio(self, eye_points):
        """Вычисление отношения высоты к ширине глаза"""
        p1, p2, p3, p4, p5, p6 = eye_points
        A = np.linalg.norm(p2 - p6)
        B = np.linalg.norm(p3 - p5)
        C = np.linalg.norm(p1 - p4)
        ear = (A + B) / (2.0 * C)
        return ear

    async def eyes_open(self,frame):
        """
        Возвращает True, если глаза открыты, иначе False.
        Использует mediapipe  face_mesh.
        """
        with self.mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True) as face_mesh:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                return False

            face = results.multi_face_landmarks[0]
            h, w, _ = frame.shape
            points = np.array([(lm.x * w, lm.y * h) for lm in face.landmark])

            left_eye =  points[self.LEFT_EYE]
            right_eye = points[self.RIGHT_EYE]

            left_ear  = self.eye_aspect_ratio(left_eye)
            right_ear = self.eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0

            # --- Порог подбирается эмпирически ---
            return ear > 0.25


class SightDetectionAsync(ClipAnalyzer):
    def __init__(self):
        super().__init__()
        self.RECT_SIZE = 20  # размер белого прямоугольника (20x20)
        self.screen = screeninfo.get_monitors()[0]
        self.model_load()
        # self.mainloop(self.cap)


    def camera_catch(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Не удалось открыть камеру.")
            raise RuntimeError("Cannot open webcam")
            return
        return cap
    
    def yolo_predicting(self, img, yolo_model='./pkgs/models/yolov11n-face.pt'):
        yolo = YOLO(yolo_model)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = yolo.predict(source=img_rgb, conf=0.35, max_det=1, imgsz=640, verbose=False)
        if len(res)==0 or len(res[0].boxes)==0:
            return None
        box = res[0].boxes[0].xyxy.cpu().numpy().astype(int).flatten()
        return box
    
    def model_load(self, main_model='./pkgs/models/model3.pth', mode='direct_screen', yolo_model='./pkgs/models/yolov11n-face.pt', execution='cuda'):
        # self.model = AFFGazeNet(output_mode=mode, pretrained=False).to('cuda')   first1_model2 model3
        self.model = AFF1(mode = 'regression').to('cuda')
        self.model.load_state_dict(torch.load(main_model, map_location=execution))
        self.model.eval()
        self.yolo = YOLO(yolo_model)
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
        return self.model, self.yolo, self.fa

    # @wrapper
    async def image_processing(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # res = self.yolo.predict(source=img_rgb, conf=0.35, max_det=1, imgsz=640, verbose=False)
        # if len(res)==0 or len(res[0].boxes)==0:
        #     return 0,0
        # box = res[0].boxes[0].xyxy.cpu().numpy().astype(int).flatten()
        box = self.yolo_predicting(img, self.yolo)
        if box is None:
            return None, None
        x1, y1, x2, y2 = box
        pad = int(0.2 * max(y2-y1, x2-x1))
        x1p = max(0, x1-pad); y1p = max(0, y1-pad)
        x2p = min(img.shape[1], x2+pad); y2p = min(img.shape[0], y2+pad)
        face_crop = img_rgb[y1p:y2p, x1p:x2p].copy()
        face_resized = cv2.resize(face_crop, (224,224))
        face_tensor = T.ToTensor()(face_resized).unsqueeze(0).float().to('cuda')
        try:
            lm_all = self.fa.get_landmarks_from_image(img_rgb)[0]
            lm_crop = lm_all - np.array([x1p, y1p])
        except Exception:
            lm_crop = None
        # yaw, pitch, roll = estimate_head_pose(lm_crop, (face_crop.shape[0], face_crop.shape[1]))
        yaw, pitch, roll = est1(lm_crop, (face_crop.shape[0], face_crop.shape[1]))
        left_eye_pts = lm_crop[36:42]
        right_eye_pts = lm_crop[42:48]
        left_eye = crop_eye(face_crop, left_eye_pts)
        right_eye = crop_eye(face_crop, right_eye_pts)
        head_pose = torch.from_numpy(np.array([yaw, pitch, roll], dtype=np.float32) / 180.0).unsqueeze(0).to('cuda')
        h, w, _ = img.shape
        x1, y1, x2, y2 = box
        bbox_norm = np.array([x1 / w, y1 / h, x2 / w, y2 / h], dtype=np.float32)
        bbox_tensor = torch.from_numpy(bbox_norm)

        left_eye_t = T.ToTensor()(left_eye).float().unsqueeze(0).to('cuda')
        right_eye_t = T.ToTensor()(right_eye).float().unsqueeze(0).to('cuda')
        bbox_tensor = bbox_tensor.unsqueeze(0).to('cuda')
        with torch.no_grad():
            out = self.model(face_tensor, left_eye_t, right_eye_t, head_pose, bbox_tensor).cpu().numpy().flatten()
        # print('Predicted normalized screen coords:', out[:2])
        screen_w, screen_h = self.screen.width, self.screen.height
        px = int(np.clip(out[0], 0, 1) * screen_w)
        py = int(np.clip(out[1], 0, 1) * screen_h)
        return px, py
        cx = px - screen_w // 2
        cy = py - screen_h // 2
        cx = int(np.clip(cx*1.5 , -screen_w//2, screen_w//2))
        cy = int(np.clip(cy*1.5,  -screen_h//2, screen_h//2))
        px_new = int(np.clip(cx + screen_w // 2, 0, screen_w))
        py_new = int(np.clip(cy + screen_h // 2, 0, screen_h))
        return px_new, py_new


    # def gaze_processing(self, cap, img):  # ON UBUNTU ONLY!!!!!!!!!
    #     img = point_on_frame(cap, img)
    #     return img
    

    def mainloop(self, cap):
        self.screen = screeninfo.get_monitors()[0]
        screen_width, screen_height = self.screen.width, self.screen.height
        print("✅ Программа запущена.")

        while True:
            frame = 255 * np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
            ret, img = cap.read()
            if not ret:
                print("⚠ Ошибка с захватом кадров")
                return
            rect_x, rect_y = self.image_processing(img)
            cv2.rectangle(frame, (rect_x, rect_y), (rect_x + self.RECT_SIZE, rect_y + self.RECT_SIZE), (255, 255, 255), -1)
            # frame = self.gaze_processing(img, frame)

            # Прямоугольник йоло-предикта
            # yolo_predict = self.yolo_predicting(img)
            # cv2.rectangle(img, yolo_predict[:2], yolo_predict[2:], (0,255,0), 2)
            
            
            # Показываем экран
            cv2.namedWindow("Testing", cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty("Testing", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Testing", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                return