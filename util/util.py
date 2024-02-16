import string
import easyocr
import paddle
import base64
import io
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

class LicensePlateReader:
    def __init__(self):
        self.easy_ocr = easyocr.Reader(['en'], gpu=False)
        self.gpu_available = paddle.device.is_compiled_with_cuda()
        print("GPU available:", self.gpu_available)
        self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)

        # ABC0000
        # ABC0S00
        # Mapping dictionaries for character conversion
        self.dict_char_to_int = {'O': '0',
                                'I': '1',
                                'J': '3',
                                'A': '4',
                                'G': '6',
                                'S': '5',
                                'B': '8',
                                'E': '6',
                                '&': '8'}

        self.dict_int_to_char = {'0': 'O',
                                '1': 'I',
                                '3': 'J',
                                '4': 'A',
                                '6': 'G',
                                '5': 'S',
                                '8': 'B'}


    def write_csv(self, results, output_path):
        """
        Write the results to a CSV file.

        Args:
            results (dict): Dictionary containing the results.
            output_path (str): Path to the output CSV file.
        """
        with open(output_path, 'w') as f:
            f.write('{},{},{},{},{},{},{},{}\n'.format('frame_nmr', 'car_id', 'car_bbox',
                                                        'license_plate_bbox', 'license_plate_bbox_score',
                                                        'license_number', 'license_number_score', 'license_plate_image'))

            for frame_nmr in results.keys():
                for car_id in results[frame_nmr].keys():
                    print("Saving results for frame: {} and car_id: {}".format(frame_nmr, car_id))
                    if 'car' in results[frame_nmr][car_id].keys() and \
                        'license_plate' in results[frame_nmr][car_id].keys() and \
                        'text' in results[frame_nmr][car_id]['license_plate'].keys():
                        f.write('{},{},{},{},{},{},{},{}\n'.format(frame_nmr,
                                                                    car_id,
                                                                    '[{} {} {} {}]'.format(
                                                                        results[frame_nmr][car_id]['car']['bbox'][0],
                                                                        results[frame_nmr][car_id]['car']['bbox'][1],
                                                                        results[frame_nmr][car_id]['car']['bbox'][2],
                                                                        results[frame_nmr][car_id]['car']['bbox'][3]),
                                                                    '[{} {} {} {}]'.format(
                                                                        results[frame_nmr][car_id]['license_plate']['bbox'][0],
                                                                        results[frame_nmr][car_id]['license_plate']['bbox'][1],
                                                                        results[frame_nmr][car_id]['license_plate']['bbox'][2],
                                                                        results[frame_nmr][car_id]['license_plate']['bbox'][3]),
                                                                    results[frame_nmr][car_id]['license_plate']['bbox_score'],
                                                                    results[frame_nmr][car_id]['license_plate']['text'],
                                                                    results[frame_nmr][car_id]['license_plate']['text_score'],
                                                                    results[frame_nmr][car_id]['license_plate']['image'])
                                )
            f.close()

    def license_complies_format(self, text):
        """
        Check if the license plate text complies with the required format.

        Args:
            text (str): License plate text.

        Returns:
            bool: True if the license plate complies with the format, False otherwise.
        """
        if len(text) != 7:
            return False

        if (text[0] in string.ascii_uppercase or text[0] in self.dict_int_to_char.keys()) and \
           (text[1] in string.ascii_uppercase or text[1] in self.dict_int_to_char.keys()) and \
           (text[2] in string.ascii_uppercase or text[2] in self.dict_int_to_char.keys()) and \
           (text[3] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[3] in self.dict_char_to_int.keys()) and \
           ((text[4] in string.ascii_uppercase or text[4] in self.dict_int_to_char.keys()) or
                (text[4] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[4] in self.dict_char_to_int.keys())) and \
           (text[5] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[5] in self.dict_char_to_int.keys()) and \
           (text[6] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] or text[6] in self.dict_char_to_int.keys()):
            return True
        else:
            return False

    def format_license(self, text):
        """
        Format the license plate text by converting characters using the mapping dictionaries.

        Args:
            text (str): License plate text.

        Returns:
            str: Formatted license plate text.
        """
        license_plate_ = ''
        mapping = {0: self.dict_int_to_char, 1: self.dict_int_to_char, 2: self.dict_int_to_char,
                    3: self.dict_char_to_int, 4: {}, 5: self.dict_char_to_int, 6: self.dict_char_to_int}
        for j in [0, 1, 2, 3, 4, 5, 6]:
            if text[j] in mapping[j].keys():
                license_plate_ += mapping[j][text[j]]
            else:
                license_plate_ += text[j]

        return license_plate_

    def read_license_plate(self, license_plate_crop, license_plate_crop_thresh, library):
        """
        Read the license plate text from the given cropped image.

        Args:
            license_plate_crop (PIL.Image.Image): Cropped image containing the license plate.

        Returns:
            tuple: Tuple containing the formatted license plate text and its confidence score.
        """

        if library == "easyocr":
            detections = self.easy_ocr.readtext(license_plate_crop_thresh)
            print("easyocr", detections)

            for detection in detections:
                bbox, text, score = detection

                text = text.upper().replace(' ', '').replace('.', '').replace('-', '')

                if self.license_complies_format(text):
                    return self.format_license(text), score

        elif library == "paddleocr":
            detections = self.paddle_ocr.ocr(license_plate_crop_thresh, cls=True)
            print("paddleocr: ", detections)
            text, score = self._get_text_and_score(detections)

            if detections == [[]] or any(len(txt) != 7 for txt in text):
                print("Invalid results, trying again without threshold")
                detections_without_thresh = self.paddle_ocr.ocr(license_plate_crop, cls=True)
                print("paddleocr: ", detections_without_thresh)
                text_without_thresh, score_without_thresh = self._get_text_and_score(detections_without_thresh)

                if (len(text_without_thresh) > len(text)) and (len(text_without_thresh) < 8):
                    print("Using results without threshold")
                    detections, text, score = detections_without_thresh, text_without_thresh, score_without_thresh
                else:
                    print("Using results with threshold")

            for detection_group in detections:
                for detection in detection_group:
                    bbox = detection[0]

                    if self.license_complies_format(text):
                        return self.format_license(text), score
                    else:
                        return text, score

        return None, None

    def _get_text_and_score(self, detections):
        """
        Extracts the text and score associated with the first detection in the provided detections.

        Args:
            detections (list): A nested list representing detections.

        Returns:
            tuple: A tuple containing the text and score associated with the first detection.
                   If no detection is found, an empty string and None are returned.
        """
        if detections != [[]]:
            text = detections[0][0][1][0]
            text = text.upper().replace(' ', '').replace('.', '').replace('-', '')
            score = detections[0][0][1][1]
            return text, score
        else:
            return '', None

    def get_car(self, license_plate, vehicle_track_ids):
        """
        Retrieve the vehicle coordinates and ID based on the license plate coordinates.

        Args:
            license_plate (tuple): Tuple containing the coordinates of the license plate (x1, y1, x2, y2, score, class_id).
            vehicle_track_ids (list): List of vehicle track IDs and their corresponding coordinates.

        Returns:
            tuple: Tuple containing the vehicle coordinates (x1, y1, x2, y2) and ID.
        """
        x1, y1, x2, y2, score, class_id = license_plate

        foundIt = False
        # Verifica todos os carros na cena
        for j in range(len(vehicle_track_ids)):
            xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]
            # Checa se a placa está dentro da bbox do carro, para fins de agregação
            if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
                car_indx = j
                foundIt = True
                break

        if foundIt:
            return vehicle_track_ids[car_indx]

        return -1, -1, -1, -1, -1

    def image_to_base64(self, image):
        """
        Convert an image to a base64 encoded string.

        Args:
            image (PIL.Image.Image): The image to be converted.

        Returns:
            str: Base64 encoded string representing the image.
        """

        image_pil = Image.fromarray(image)
        buffered = io.BytesIO()
        image_pil.save(buffered, format="JPEG")

        return base64.b64encode(buffered.getvalue()).decode("utf-8")

