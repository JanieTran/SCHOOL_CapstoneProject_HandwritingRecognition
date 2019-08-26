import tensorflow as tf
import utils
import Levenshtein
from DataManager import DataManager, resize_image


with tf.gfile.GFile('./checkpoint/inverse/frozen_model_inverted.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name='CRNN')

x = graph.get_tensor_by_name('CRNN/inputs:0')
y = graph.get_tensor_by_name('CRNN/CTCBeamSearchDecoder:1')
seq_len = graph.get_tensor_by_name('CRNN/sequence_length:0')


def recognise(inputs):
    with tf.Session(graph=graph) as sess:
        y_out = sess.run(y, feed_dict={
            x: inputs,
            seq_len: [127]
        })

    return utils.decode_result(y_out)


def calculate_accuracy(original_seq, decoded_seq):
    if len(original_seq) <= len(decoded_seq):
        decoded = [decoded_seq[i] for i in range(len(original_seq))]
    else:
        decoded = [''] * len(original_seq)
        for i in range(len(decoded_seq)):
            decoded[i] = decoded_seq[i]

    count = 0
    # For each character in original sequence
    for e in range(len(original_seq)):
        if original_seq[e] == decoded[e]:
            count += 1

    return count * 1.0 / len(original_seq)


avg_distance = 0
avg_accuracy = 0

print('Loading data...')
train_feeder = DataManager(train=True)
print('Train size:', train_feeder.size)

for i in range(train_feeder.size):
    if i % 100 == 0:
        print(i)

    _, batch_inputs, batch_text, _ = train_feeder.generate_batch(index=[i])
    predicted = recognise(batch_inputs)

    accuracy = calculate_accuracy(original_seq=batch_text[0], decoded_seq=predicted)
    distance = Levenshtein.distance(batch_text[0], predicted)

    avg_accuracy += accuracy
    avg_distance += distance

print('Average accuracy:', avg_accuracy / train_feeder.size)
print('Average distance:', avg_distance / train_feeder.size)
