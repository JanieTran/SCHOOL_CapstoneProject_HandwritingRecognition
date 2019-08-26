import zmq


context = zmq.Context()
socket = context.socket(zmq.REP)

port = '8181'
socket.bind(f'tcp://192.168.137.1:{port}')

print('Starting ZMQ server to receive labels..')

while True:
    print('\nListening for labels...')
    message = socket.recv()
    print('Receiving label...')

    label = message.decode('utf-8')
    print(label)

    with open('application/collected.txt', 'a') as collected_file:
        collected_file.write(',{}'.format(label))

    socket.send_string('Label received.')

socket.close()
context.term()
