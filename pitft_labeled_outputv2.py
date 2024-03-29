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
import keyboard  # using module keyboard
from scipy.spatial import distance as dist
from collections import OrderedDict 
from operator import getitem 
import mysql.connector
from mysql.connector import errorcode
import datetime   

MODEL_FILENAME_face = '/home/pi/Documents/rpiformulisa/includes/model_face.tflite'
LABELS_FILENAME_face = '/home/pi/Documents/rpiformulisa/includes/labels_face.txt'
MODEL_FILENAME_cars = '/home/pi/Documents/rpiformulisa/includes/model_cars.tflite'
LABELS_FILENAME_cars = '/home/pi/Documents/rpiformulisa/includes/labels_cars.txt'

# Obtain connection string information from the portal
config = {
  'host':'mysqlformulisa.mysql.database.azure.com',
  'user':'bmiranda@mysqlformulisa',
  'password':'formulisa..,2019',
  'database':'rpiformulisa'
}


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
#screen = pygame.display.set_mode((0,0), pygame.FULLSCREEN)

#capture_manager = PiCameraStream(resolution=(screen.get_width(), screen.get_height()), rotation=180, preview=False)
capture_manager = PiCameraStream(resolution=(320, 240), rotation=180, preview=False)

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



def main(opt, model, smallfont, medfont, bigfont, centroid_base, id_key, ct_frame):
    global last_spoken
    temp_new_cars = 0
    
    if opt == 3:
        # Construct connection string
        try:
           conn = mysql.connector.connect(**config)
           print("Conexion establecida a Base de datos")
        except mysql.connector.Error as err:
          if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
            print("Ocurrio un error con el user name or password")
          elif err.errno == errorcode.ER_BAD_DB_ERROR:
            print("Base de datos no existe")
          else:
            print(err)
        else:
          cursor = conn.cursor()
    
    while not capture_manager.stopped:
        if capture_manager.frame is None:
            continue
        frame = capture_manager.read()
        # get the raw data frame & swap red & blue channels
        previewframe = np.ascontiguousarray(np.flip(np.array(capture_manager.frame), 2))
        # make it an image
        img = pygame.image.frombuffer(previewframe, capture_manager.camera.resolution, 'RGB')
        # draw it!
        #screen.blit(img, (0, 0))

        timestamp = time.monotonic()
        if opt == 1 or opt==3:
            img2 = Image.fromarray(frame)
            prediction = model.predict_image(img2)
        if opt == 2:
            prediction = model.tflite_predict(frame)[0]
        logging.info(prediction)
        delta = time.monotonic() - timestamp
        logging.info("%s inference took %d ms, %0.1f FPS" % ("TFLite", delta * 1000, 1 / delta))
        print(last_seen)

        # add FPS on top corner of image
        #fpstext = "%0.1f FPS" % (1/delta,)
        #fpstext_surface = smallfont.render(fpstext, True, (255, 0, 0))
        #fpstext_position = (screen.get_width()-10, 10) # near the top right corner
        #screen.blit(fpstext_surface, fpstext_surface.get_rect(topright=fpstext_position))
        if keyboard.is_pressed('q') or keyboard.is_pressed('a'):
            print("saliendo del modelo\n seleccione opcion:\n q: Deteccion caras\n a: Deteccion OBjetos\n")
            break

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
            new_cars = len(centroid_base) - temp_new_cars
            if new_cars>0:
                temp_new_cars = len(centroid_base)
                print('cantidad autos nuevos: '+str(new_cars))
                # Insert quantity new cars and date
                try:
                    cursor.execute("INSERT INTO carsformulisa (quantity, date) VALUES (%s, %s);", (new_cars, datetime.datetime.now()))
                    print("Inserted",cursor.rowcount,"row(s) of data.")  
                except mysql.connector.Error:
                    conn = mysql.connector.connect(**config)
                    print("Conexion establecida a Base de datos")
                    cursor = conn.cursor()
                    cursor.execute("INSERT INTO carsformulisa (quantity, date) VALUES (%s, %s);", (new_cars, datetime.datetime.now()))
                    print("Inserted",cursor.rowcount,"row(s) of data.")  
            print(centroid_base)
            print('cantidad autos: '+str(len(centroid_base)))  
            print('ct_frama: '+str(ct_frame))
            ct_frame = ct_frame + 1       
            conn.commit()   

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
                        if detectsize[0] < 320: # it'll fit!
                            detecttextfont = f
                            break
                    else:
                        detecttextfont = smallfont # well, we'll do our best
                    detecttext_color = (0, 255, 0) if persistant_obj else (255, 255, 255)
                    detecttext_surface = detecttextfont.render(detecttext, True, detecttext_color)
                    #detecttext_position = (screen.get_width()//2,
                    #                       screen.get_height() - detecttextfont.size(detecttext)[1])
                    #screen.blit(detecttext_surface, detecttext_surface.get_rect(center=detecttext_position))

                    if persistant_obj and last_spoken != detecttext:
                        os.system('echo %s | festival --tts & ' % detecttext)
                        last_spoken = detecttext
                    break
                
            else:
                last_seen.append(None)
                last_seen.pop(0)
                if last_seen.count(None) == len(last_seen):
                    last_spoken = None

        #pygame.display.update()

def insert_text(input_text, opt):
    if opt==1:
        fpstext_surface = smallfont.render(input_text, True, (255, 255, 0))
        fpstext_position = (screen.get_width()//2,screen.get_height()//2) 
        screen.blit(fpstext_surface, fpstext_surface.get_rect(center=fpstext_position))
    else:
        fpstext_surface = smallfont.render(input_text, True, (0, 255, 0))
        fpstext_position = (screen.get_width()//2, screen.get_height()//3) 
        screen.blit(fpstext_surface, fpstext_surface.get_rect(center=fpstext_position))

def run_process():
    print("Cargando modelos en memoria, por favor espere...")
    old_switch_q = False
    old_switch_a = False
    old_switch_z = False
    centroid_base = {}
    id_key = 1
    ct_frame = 0

    #pygame.mouse.set_visible(False)
    #screen.fill((0,0,0))
    try:
        splash = pygame.image.load(os.path.dirname(sys.argv[0])+'/screen3.jpg')
        #screen.blit(splash, ((screen.get_width() / 2) - (splash.get_width() / 2),
        #            (screen.get_height() / 2) - (splash.get_height() / 2)))
    except pygame.error:
        pass
    #pygame.display.update()

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
    model = MobileNetV2Base(include_top=True)
    
    capture_manager.start()
    
    print("Carga de modelos lista, seleccione opcion:\n q: Deteccion caras\n a: Deteccion OBjetos\n")

    while True:  # making a loop
        new_switch_q = keyboard.is_pressed('q')
        new_switch_a = keyboard.is_pressed('a')
        new_switch_z = keyboard.is_pressed('z')
        if new_switch_q==True and old_switch_q==False:  # if key 'q' is pressed 
            print('Iniciando Modelo Deteccion de Caras, por favor espere!')
            main(1, face_model, smallfont, medfont, bigfont, centroid_base, id_key, ct_frame)
            time.sleep(0.1)
        if new_switch_a==True and old_switch_a==False:  # if key 'q' is pressed 
            print('Iniciando Modelo Deteccion de OBjetos, por favor espere!')
            main(2, model, smallfont, medfont, bigfont, centroid_base, id_key, ct_frame)
            time.sleep(0.1)
        if new_switch_z==True and old_switch_z==False:  # if key 'q' is pressed 
            print('Iniciando Modelo Deteccion de autos, por favor espere!')
            main(3, cars_model, smallfont, medfont, bigfont, centroid_base, id_key, ct_frame)
            time.sleep(0.1)
        old_switch_q = new_switch_q
        old_switch_a = new_switch_a
        old_switch_z = new_switch_z
    

if __name__ == "__main__":
    #args = parse_args()
    try:
        #main(args)
        run_process()
    except KeyboardInterrupt:
        capture_manager.stop()
