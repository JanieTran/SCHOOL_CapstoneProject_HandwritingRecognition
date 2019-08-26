import zmq
import io
import matplotlib.pyplot as plt

from PIL import Image
from ImagePreprocessing import ImagePreprocessing
from FinalModel import FinalModel
from datetime import datetime


context = zmq.Context()
socket = context.socket(zmq.REP)

port = '8080'
socket.bind(f'tcp://192.168.137.1:{port}')

print('Starting ZMQ server to receive images..')

image_preprocessor = ImagePreprocessing()
final_model = FinalModel()

while True:
    print('\nListening for images...')
    message = socket.recv()
    print('Receiving image...')

    raw_im = Image.open(io.BytesIO(message)).convert(mode='L')
    raw_im = raw_im.transpose(Image.ROTATE_90)
    raw_im.save('raw.png')

    prep_im = image_preprocessor.enhance_and_filter(image=raw_im, brightness=1.4, contrast=1.2)
    filename = datetime.now().strftime('%Y%m%d_%H%M%S')
    prep_im.save('application/{}.png'.format(filename))

    with open('application/collected.txt', 'a') as collected_file:
        collected_file.write('\n{}'.format(filename))

    prediction = final_model.run(image=prep_im)
    print('Prediction:', prediction)

    plt.title(prediction)
    plt.imshow(prep_im, cmap='gray')
    plt.show()

    socket.send_string(prediction)

socket.close()
context.term()
