import tensorflow as tf
import numpy as np
import os
from txt_handler import DataGenerator
from rnn_model import RNNSeqModel


LEARNING_RATE = 1e-4
KEEP_PROB = .8
NUM_LAYERS = 3
NUM_HIDDEN_UNITS = 512 # num units in each lstm-layer
BATCH_SIZE = 128
SEQ_LENGTH = 40
NUM_EPOCHS = 10*330
DATA_PATH = 'shakespeare_data'
LOG_DIR = './log_dir'
TXT_OUT_DIR = './model_txt_out'
MODEL_DIR = './model_dir'
RESTORE_FROM_CKPT = False

def main():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

    if not os.path.exists(TXT_OUT_DIR):
        os.makedirs(TXT_OUT_DIR)

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    #object that decodes/encodes data and provides training-batches
    txt_pipeline = DataGenerator(data_path=DATA_PATH)
    with tf.Session() as sess:
        model = RNNSeqModel(lr=LEARNING_RATE,
                            num_chars=txt_pipeline.num_chars,
                            num_layers=NUM_LAYERS,
                            num_hidden=NUM_HIDDEN_UNITS)
        sess.run(tf.global_variables_initializer())
        train_loop(sess, model, txt_pipeline)

def train_loop(sess, model, txt_pipeline):
    writer = tf.summary.FileWriter(LOG_DIR)
    writer.add_graph(sess.graph)
    saver = tf.train.Saver(max_to_keep=5)
    if RESTORE_FROM_CKPT:
        ckpt = tf.train.get_checkpoint_state(MODEL_DIR)
        print('restore ckpt: ', ckpt)
        saver.restore(sess, ckpt.model_checkpoint_path)
    # initial state of the LSTM-stack, 2 is for hidden states, i.e. c and h
    lstm_state_init = np.zeros([NUM_LAYERS, 2, BATCH_SIZE, NUM_HIDDEN_UNITS])

    train_steps = 0
    num_chars = txt_pipeline.num_chars

    for x, y, epoch, reset_state in txt_pipeline.batch_generator(BATCH_SIZE, SEQ_LENGTH, NUM_EPOCHS):

        # reset LSTM-state at the start of each epoch
        if reset_state:
            lstm_state = lstm_state_init

        train_steps +=1
        feed_dict= {model.placeholders['lstm_state']: lstm_state,
                    model.placeholders['x']:x,
                    model.placeholders['y']:y,
                    model.placeholders['keep_prob']: KEEP_PROB,
                    }
        # write to tensorboard every 20th update
        if not train_steps % 20 == 0:
            lstm_state, _ = sess.run([model.output['lstm_state_out'], model.update], feed_dict)
        else:
            lstm_state, loss, _ = sess.run([model.output['lstm_state_out'], model.loss, model.update], feed_dict)
            train_summary = tf.Summary()
            train_summary.value.add(simple_value=float(loss), tag="X_ent")
            writer.add_summary(train_summary, train_steps)

        # sometimes generate a sequence of text for fun/evaluation
        if train_steps % 2000 == 0:
            encoded_txt = []
            x_ = np.random.randint(0, num_chars - 1)
            encoded_txt.append(x_)
            lstm_state_ =  np.zeros([NUM_LAYERS, 2, 1, NUM_HIDDEN_UNITS])
            for _ in range(1500):
                feed_dict = {model.placeholders['lstm_state']: lstm_state_,
                             model.placeholders['x']: np.reshape(x_, [-1, 1]),
                             model.placeholders['keep_prob']: 1.
                             }
                cat_dist, lstm_state_ = sess.run([model.output['cat_dist'], model.output['lstm_state_out']], feed_dict)
                # sample character from categorical distribution
                x_ = np.random.choice(num_chars,size=1, p=np.reshape(cat_dist, [-1]))[0]
                encoded_txt.append(x_)

            text = txt_pipeline.decode_data(encoded_txt)# from list of integers to string

            with open(TXT_OUT_DIR + '/model_out_' + str(train_steps), 'w') as txt_file:
                txt_file.write(text)

        if train_steps % 1000 == 0:
            saver.save(sess, MODEL_DIR + '/model-'+ str(train_steps) +'.ckpt')

#Todo: inference

if __name__ == '__main__':
    main()