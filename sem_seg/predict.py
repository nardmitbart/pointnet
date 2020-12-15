import argparse
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
from model import *
import provider

# see: https://stackoverflow.com/questions/11125878/python-output-to-file-higher-floating-point-precision
class myFloat( float ):
    def __str__(self):
        return "%.6f"%self

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
FLAGS = parser.parse_args()

# read flags or deafault values
BATCH_SIZE = 1
DATA_PATH = '/home/felix/Desktop/prj/data/test_neu/test.h5'
EXPORT_PATH = '/home/felix/Desktop/prj/data/test_neu/prediction.csv'
MODEL_PATH = '/home/felix/Desktop/git/pointnet/sem_seg/log6/model.ckpt'
DUMP_DIR = '/home/felix/Desktop/git/pointnet/sem_seg/dump/'
OUTPUT_FILELIST = '/home/felix/Desktop/git/pointnet/sem_seg/log6/output_filelist.txt'
GPU_INDEX = FLAGS.gpu

NUM_POINT = 2048
NUM_FEATURES = 3
NUM_CLASSES = 2

if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

# data_batch: (num_batches, batchsize, num_features) zb. (81, 2048, 3)
# label_batch: (num_batches, batchsize) zb. (81, 2048)
data_batch, label_batch = provider.loadDataFile(DATA_PATH)

NUM_BATCHES = data_batch.shape[0]

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def export(predicted_labels, path):
    dump = []

    for i in range(NUM_BATCHES):
        for j in range(2048):
            dump.append((
                myFloat(data_batch[i][j][0]), 
                myFloat(data_batch[i][j][1]),
                myFloat(data_batch[i][j][2]),
                predicted_labels[i][j]))

    npdump = np.asarray(dump, dtype=np.float64)

    fmt = '%1.8f', '%1.8f', '%1.8f', '%d'
    np.savetxt(path, npdump, delimiter=' ', fmt=fmt)

    print('exported prediction to: ' + EXPORT_PATH)


def evaluate():

    with tf.device('/gpu:'+str(GPU_INDEX)):
        pointclouds_pl, labels_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT, NUM_FEATURES)
        is_training_pl = tf.placeholder(tf.bool, shape=())

        # simple model
        pred = get_model(pointclouds_pl, is_training_pl, NUM_FEATURES, NUM_CLASSES)
        loss = get_loss(pred, labels_pl)
        pred_softmax = tf.nn.softmax(pred)
 
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = True
    sess = tf.Session(config=config)

    # Restore variables from disk.
    saver.restore(sess, MODEL_PATH)
    log_string("Model restored.")

    ops = {'pointclouds_pl': pointclouds_pl,
           'labels_pl': labels_pl,
           'is_training_pl': is_training_pl,
           'pred': pred,
           'pred_softmax': pred_softmax,
           'loss': loss}
    
    predicted_labels = eval_one_epoch(sess, ops)

    # TODO: export proba too
    export(predicted_labels, EXPORT_PATH)


def eval_one_epoch(sess, ops):
    error_cnt = 0
    is_training = True
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    # TODO: maybe provider.shuffle_data
    # be careful when exporting results to csv !!
    current_data = data_batch
    current_label = label_batch

    pred_labels = []

    for batch_idx in range(NUM_BATCHES):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        
        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training}

        loss_val, pred_val = sess.run([ops['loss'], ops['pred_softmax']], feed_dict=feed_dict)

        # TODO: maybe no clutter: only classify points with a high probability
        pred_label = np.argmax(pred_val, axis=2) # BxN

        pred_labels.append(pred_label)

        correct = np.sum(pred_label == current_label[start_idx:end_idx,:])
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += (loss_val*BATCH_SIZE)

        for i in range(start_idx, end_idx):
            for j in range(NUM_POINT):
                l = current_label[i, j]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_label[i-start_idx, j] == l)

    log_string('eval mean loss: %f' % (loss_sum / float(total_seen/NUM_POINT)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))

    pred_labels = np.asarray(pred_labels, dtype=np.int8)
    pred_labels = np.squeeze(pred_labels)

    return pred_labels


if __name__=='__main__':
    with tf.Graph().as_default():
        evaluate()
    LOG_FOUT.close()
