# Python
import time
import logging
import argparse
import pygame
import os
import sys
import numpy as np
import subprocess
from object_detection import ObjectDetection
import tensorflow as tf
import PIL.Image as Image
import keyboard 
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BCM)

GPIO.setup(27, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(23, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(22, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(17, GPIO.IN, pull_up_down=GPIO.PUD_UP)

old_27 = 1
old_23 = 1
old_22 = 1
old_17 = 1
black = (0,0,0)

MODEL_FILENAME_face = 'includes/model_face.tflite'
LABELS_FILENAME_face = 'includes/labels_face.txt'
MODEL_FILENAME_cars = 'includes/model_cars.tflite'
LABELS_FILENAME_cars = 'includes/labels_cars.txt'

CONFIDENCE_THRESHOLD = 0.5   # at what confidence level do we say we detected a thing
PERSISTANCE_THRESHOLD = 0.25  # what percentage of the time we have to have seen a thing

os.environ['SDL_FBDEV'] = "/dev/fb1"
os.environ['SDL_VIDEODRIVER'] = "fbcon"

# App
from rpi_vision.agent.capture import PiCameraStream
from rpi_vision.models.mobilenet_v2 import MobileNetV2Base

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

# initialize the display
pygame.init()
screen = pygame.display.set_mode((0,0), pygame.FULLSCREEN)

capture_manager = PiCameraStream(resolution=(screen.get_width(), screen.get_height()), rotation=180, preview=False)
#capture_manager = PiCameraStream(resolution=(320, 240), rotation=180, preview=False)

class TFLiteObjectDetection(ObjectDetection):
    """Object Detection class for TensorFlow Lite"""
    def __init__(self, model_filename, labels):
        super(TFLiteObjectDetection, self).__init__(labels)
        self.interpreter = tf.lite.Interpreter(model_path=model_filename)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]['index']
        self.output_index = self.interpreter.get_output_details()[0]['index']

    def predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=np.float32)[np.newaxis, :, :, (2, 1, 0)]  # RGB -> BGR and add 1 dimension.

        # Resize input tensor and re-allocate the tensors.
        self.interpreter.resize_tensor_input(self.input_index, inputs.shape)
        self.interpreter.allocate_tensors()
        
        self.interpreter.set_tensor(self.input_index, inputs)
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_index)[0]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--include-top', type=bool,
                        dest='include_top', default=True,
                        help='Include fully-connected layer at the top of the network.')

    parser.add_argument('--tflite',
                        dest='tflite', action='store_true', default=False,
                        help='Convert base model to TFLite FlatBuffer, then load model into TFLite Python Interpreter')
    args = parser.parse_args()
    return args

last_seen = [None] * 10
last_spoken = None

def main(args, opt, model, smallfont, medfont, bigfont, centroid_base, id_key, ct_frame):
    pygame.display.update()
    global last_spoken
    
    while not capture_manager.stopped:
        if capture_manager.frame is None:
            continue
        frame = capture_manager.read()
        # get the raw data frame & swap red & blue channels
        previewframe = np.ascontiguousarray(np.flip(np.array(capture_manager.frame), 2))
        # make it an image
        img = pygame.image.frombuffer(previewframe, capture_manager.camera.resolution, 'RGB')
        # draw it!
        screen.blit(img, (0, 0))

        timestamp = time.monotonic()
        if not(GPIO.input(27)):
            t_menu = "Seleccione Opción:\n 17: Detección de caras\n 22: Detección de Objetos\n 23: Conteo de Autos\n 27: Volver al Menu\n"
            pygame.display.update()
            screen.fill(black)
            insert_text(t_menu, 1)
            break
            
        if args.tflite:
            if opt == 1 or opt == 3:
                img2 = Image.fromarray(frame)
                prediction = model.predict_image(img2)
            if opt == 2:
                prediction = model.tflite_predict(frame)[0]
        else:
            prediction = model.predict(frame)[0]
        logging.info(prediction)
        delta = time.monotonic() - timestamp
        logging.info("%s inference took %d ms, %0.1f FPS" % ("TFLite" if args.tflite else "TF", delta * 1000, 1 / delta))
        print(last_seen)

        # add FPS on top corner of image
        fpstext = "%0.1f FPS" % (1/delta,)
        fpstext_surface = smallfont.render(fpstext, True, (255, 0, 0))
        fpstext_position = (screen.get_width()-10, 10) # near the top right corner
        screen.blit(fpstext_surface, fpstext_surface.get_rect(topright=fpstext_position))

        if opt==3:
            centroid_temp = {}
            print(prediction)
            for p in prediction:
                print('probabilidad: '+str(p['probability']))
                if p['probability'] > 0.42:
                    start_x = p['boundingBox']['left']
                    start_y = p['boundingBox']['top']
                    end_x = p['boundingBox']['width']
                    end_y = p['boundingBox']['height']
                    cx = (start_x + end_x)/2.0
                    cy = (start_y + end_y)/2.0
                    centroid_temp[id_key] = {}
                    centroid_temp[id_key]['cx'] = cx
                    centroid_temp[id_key]['cy'] = cy
                    centroid_temp[id_key]['q'] = ct_frame
                    id_key = id_key + 1
            if len(centroid_base) == 0 and len(centroid_temp)==0:
                print('continue')
            elif len(centroid_base) == 0 and len(centroid_temp)>0:
                print('carga base')
                centroid_base = centroid_temp
            else:
                for p_id, p_info in centroid_temp.items():	
                    a = np.array([[p_info['cx'],p_info['cy']]])
                    l_dist = {}
                    for c_id, c_info in centroid_base.items():	
                        b = np.array([[c_info['cx'],c_info['cy']]])
                        D = dist.cdist(a, b)
                        l_dist[c_id] = {}
                        l_dist[c_id]['d'] = D[0][0]
                        l_dist[c_id]['cx'] = p_info['cx']
                        l_dist[c_id]['cy'] = p_info['cy']
                        #l_dist.append(D[0][0])
                    print('lista')
                    res = OrderedDict(sorted(l_dist.items(), key = lambda x: getitem(x[1], 'd'))) 
                    print(res)   
                    first_key = list(res.keys())[0]
                    print(first_key)
                    check_new = centroid_base[first_key]['q']+2
                    if res[first_key]['d']<0.05 and ct_frame<check_new:
                        print('mismo')
                        centroid_base[first_key]['cx'] = res[first_key]['cx']
                        centroid_base[first_key]['cy'] = res[first_key]['cy']
                        centroid_base[first_key]['q'] = ct_frame
                    else:
                        print('nuevo')
                        centroid_base[p_id] = {}
                        centroid_base[p_id]['cx'] = p_info['cx']
                        centroid_base[p_id]['cy'] = p_info['cy']
                        centroid_base[p_id]['q'] = ct_frame
            print(centroid_base)
            print('cantidad autos: '+str(len(centroid_base)))  
            cant_cars = len(centroid_base)
            cartext = "Cantidad de Autos: %0.0f " % cant_cars
            cartext_surface = smallfont.render(cartext, True, (0, 255, 0))
            cartext_position = (screen.get_width()//2, screen.get_height()//3) 
            screen.blit(cartext_surface, cartext_surface.get_rect(center=cartext_position))
            print('ct_frama: '+str(ct_frame))
            ct_frame = ct_frame + 1          

        else:
            for p in prediction:
                if opt == 1:
                    name = p['tagName']
                    conf = p['probability']
                if opt == 2:
                    label, name, conf = p
                if conf > CONFIDENCE_THRESHOLD:
                    print("Detected", name)

                    persistant_obj = False  # assume the object is not persistant
                    last_seen.append(name)
                    last_seen.pop(0)

                    inferred_times = last_seen.count(name)
                    if inferred_times / len(last_seen) > PERSISTANCE_THRESHOLD:  # over quarter time
                        persistant_obj = True
                    
                    detecttext = name.replace("_", " ")
                    detecttextfont = None
                    for f in (bigfont, medfont, smallfont):
                        detectsize = f.size(detecttext)
                        if detectsize[0] < screen.get_width(): # it'll fit!
                            detecttextfont = f
                            break
                    else:
                        detecttextfont = smallfont # well, we'll do our best
                    detecttext_color = (0, 255, 0) if persistant_obj else (255, 255, 255)
                    detecttext_surface = detecttextfont.render(detecttext, True, detecttext_color)
                    detecttext_position = (screen.get_width()//2,
                                           screen.get_height() - detecttextfont.size(detecttext)[1])
                    screen.blit(detecttext_surface, detecttext_surface.get_rect(center=detecttext_position))

                    if persistant_obj and last_spoken != detecttext:
                        os.system('echo %s | festival --tts & ' % detecttext)
                        last_spoken = detecttext
                    break
                
            else:
                last_seen.append(None)
                last_seen.pop(0)
                if last_seen.count(None) == len(last_seen):
                    last_spoken = None

        pygame.display.update()
     
def insert_text(input_text, opt):
    if opt==1:
        fpstext_surface = smallfont.render(input_text, True, (255, 255, 255))
        fpstext_position = (screen.get_width()//2,screen.get_height()//2) 
        screen.blit(fpstext_surface, fpstext_surface.get_rect(center=fpstext_position))
    else:
        fpstext_surface = smallfont.render(input_text, True, (0, 255, 0))
        fpstext_position = (screen.get_width()//2, screen.get_height()//3) 
        screen.blit(fpstext_surface, fpstext_surface.get_rect(center=fpstext_position))     
        
def run_process():
    t_load_model = "Cargando modelos en memoria, por favor espere..."

    pygame.mouse.set_visible(False)
    screen.fill(black)
    insert_text(t_load_model, 1)

    # use the default font
    smallfont = pygame.font.Font(None, 24)
    medfont = pygame.font.Font(None, 36)
    bigfont = pygame.font.Font(None, 48)

    #model face
    with open(LABELS_FILENAME_face, 'r') as f:
        labels_face = [l.strip() for l in f.readlines()]
    face_model = TFLiteObjectDetection(MODEL_FILENAME_face, labels_face)
    
    #model cars
    with open(LABELS_FILENAME_cars, 'r') as f:
        labels_cars = [l.strip() for l in f.readlines()]
    cars_model = TFLiteObjectDetection(MODEL_FILENAME_cars, labels_cars)
    
    #model objects
    model = MobileNetV2Base(include_top=args.include_top)
    
    capture_manager.start()
    
    t_menu = "Seleccione Opción:\n 17: Detección de caras\n 22: Detección de Objetos\n 23: Conteo de Autos\n 27: Volver al Menu\n"
    pygame.display.update()
    screen.fill(black)
    insert_text(t_menu, 1)

    while True:  # making a loop
        new_27 = GPIO.input(27)
        new_23 = GPIO.input(23)
        new_22 = GPIO.input(22)
        new_17 = GPIO.input(17)
        if not(new_27) and old_27 == 1:
            t_menu = "Seleccione Opción:\n 17: Detección de caras\n 22: Detección de Objetos\n 23: Conteo de Autos\n 27: Volver al Menu\n"
            pygame.display.update()
            screen.fill(black)
            insert_text(t_menu, 1)
            time.sleep(0.1)
        if not(new_23) and old_23 == 1:
            t_model = "Iniciando Modelo Detección de Autos, por favor espere!"
            pygame.display.update()
            screen.fill(black)
            insert_text(t_model, 1)
            main(args, 3, cars_model, smallfont, medfont, bigfont, centroid_base, id_key, ct_frame)
            time.sleep(0.1)
        if not(new_22) and old_22 == 1:
            t_model = "Iniciando Modelo Detección de Objetos, por favor espere!"
            pygame.display.update()
            screen.fill(black)
            insert_text(t_model, 1)
            main(args, 2, model, smallfont, medfont, bigfont, centroid_base, id_key, ct_frame)
            time.sleep(0.1)
        if not(new_17) and old_17 == 1:
            t_model = "Iniciando Modelo Detección de Caras, por favor espere!"
            pygame.display.update()
            screen.fill(black)
            insert_text(t_model, 1)
            main(args, 1, face_model, smallfont, medfont, bigfont, centroid_base, id_key, ct_frame)
            time.sleep(0.1)
        old_27 = new_27
        old_23 = new_23
        old_22 = new_22
        old_17 = new_17
    GPIO.cleanup()
    

if __name__ == "__main__":
    args = parse_args()
    try:
        #main(args)
        run_process()
    except KeyboardInterrupt:
        capture_manager.stop()
