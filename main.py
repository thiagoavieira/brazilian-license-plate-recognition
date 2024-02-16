import cv2
import numpy as np
import PIL
from PIL import Image
from ultralytics import YOLO
from sort.sort import *
from util.util import LicensePlateReader

class LicensePlateProcessor:
    def __init__(self):
        self.license_plate_reader = LicensePlateReader()
        self.results = {}
        self.invalid_results = {}

        # Inicializando o rastreador de objetos
        self.mot_tracker = Sort()
        
        # Carregando o modelo COCO para detecção de veículos
        self.coco_model = YOLO('./models/yolov8m.pt')

        # Carregando o modelo YOLO treinado para detecção de placas de carro
        # e de acordo com um funcionário da ultralytics:
        # This approach seems a bit unorthodox but ensures a level of compatibility and
        # avoids certain types of issues related to dependencies and environment setup
        self.license_plate_detector = YOLO('./models/yolov8m.pt')
        self.license_plate_detector = YOLO('./models/second-train/weights/best.pt')

        # Lista de IDs de classes para veículos ('car', 'motorcycle', 'bus', 'truck')
        self.vehicles = [2, 3, 5, 7]

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_nmr = -1
        ret = True
        while ret:
            frame_nmr += 1
            ret, frame = cap.read()
            if ret:
                # if frame_nmr > 50:  # Ajustado para os primeiros 10 frames
                #     break
                # uma chave para cada frame
                self.results[frame_nmr] = {}
                self.invalid_results[frame_nmr] = {}
                self.process_frame(frame, frame_nmr)
        cap.release()

        return self.results, self.invalid_results

    def process_frame(self, frame, frame_nmr):
        # Detectando veículos no frame
        detections = self.coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in self.vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # Rastreando veículos
        track_ids = self.mot_tracker.update(np.asarray(detections_))

        # Detectando placas de carro no frame
        license_plates = self.license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            
            # Atribuindo a placa ao veículo
            xcar1, ycar1, xcar2, ycar2, car_id = self.license_plate_reader.get_car(license_plate, track_ids)

            if car_id != -1:
                # Cortando a região da placa do veículo
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                # Processando a placa para grayscale e aplicando threshold
                license_plate_crop_thresh = self.apply_first_threshold(license_plate_crop)

                # Lendo os números da placa
                license_plate_text, license_plate_text_score = self.license_plate_reader.read_license_plate(license_plate_crop, license_plate_crop_thresh, "paddleocr")

                # Salvando os resultados
                results_dict = {
                    'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]}, 
                    'license_plate': {
                        'bbox': [x1, y1, x2, y2], 
                        'text': license_plate_text, 
                        'bbox_score': score, 
                        'text_score': license_plate_text_score,
                        'image': self.license_plate_reader.image_to_base64(license_plate_crop)
                    }
                }

                if license_plate_text is not None:
                    # Verifica se o texto da placa tem 7 caracteres e se a pontuação é válida
                    if (len(license_plate_text) == 7) and (license_plate_text_score is not None):
                        self.results[frame_nmr][car_id] = results_dict
                    else:
                        self.invalid_results[frame_nmr][car_id] = results_dict
                else:
                    self.invalid_results[frame_nmr][car_id] = results_dict
                
    
    def apply_first_threshold(self, license_plate):
        """
        Apply the first threshold to the license plate image.

        Args:
            license_plate (numpy.ndarray): Image of the license plate.

        Returns:
            numpy.ndarray: License plate image after applying the threshold.

        Description:
            This function applies a threshold to the license plate image, where pixels with intensity less than 64 are set to 0 (black), 
            and pixels with intensity greater than or equal to 64 are set to 255 (white). 
            cv2.THRESH_BINARY_INV means that values above the threshold will be set to 0 and values below the threshold will be set to 255 (inverse).
        """
        license_plate_crop_gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
        _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

        # Se quiser ver o threshold que foi aplicado na placa é só descomentar essas 3 linhas abaixo
        # cv2.imshow('original_crop', license_plate)
        # cv2.imshow('threshold', license_plate_crop_thresh)
        # cv2.waitKey(0)

        return license_plate_crop_thresh
    
    def apply_second_threshold(self, license_plate):
        """
        Apply the second type of image processing to the license plate image.

        Args:
            license_plate (numpy.ndarray): Image of the license plate.

        Returns:
            numpy.ndarray: License plate image after applying the threshold.

        Description:
            This function applies a second type of image processing to the license plate image. 
            It first converts the image to grayscale and then applies Gaussian blur to reduce noise.
            After that, it applies Otsu's thresholding method to automatically calculate the optimal threshold value.
            The result is a binary image where values above the threshold are set to 0 and values below the threshold are set to 255 (inverse).
            Finally, it removes noise from the image and inverts it. 
            If you want to visualize the intermediate steps of the thresholding process, uncomment the corresponding lines in the code.
        """
        license_plate_crop_gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(license_plate_crop_gray, (3,3), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Removendo ruido e invertendo a imagem
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        license_plate_crop_thresh = 255 - opening

        # Se quiser ver o threshold que foi aplicado na placa é só descomentar essas 4 linhas abaixo
        # cv2.imshow('thresh', thresh)
        # cv2.imshow('opening', opening)
        # cv2.imshow('invert', license_plate_crop_thresh)
        # cv2.waitKey()

        return license_plate_crop_thresh
    
    def main(self):
        
        # Processando o vídeo
        results, invalid_results = self.process_video('./videos/transito.mp4')

        # Escrevendo os resultados em um arquivo CSV
        self.license_plate_reader.write_csv(results, './results/results.csv')
        self.license_plate_reader.write_csv(invalid_results, './results/invalid_results.csv')


if __name__ == "__main__":
    processor_instance = LicensePlateProcessor()
    processor_instance.main()