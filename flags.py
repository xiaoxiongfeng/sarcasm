import tensorflow as tf
import os

flags = tf.flags
FLAGS = flags.FLAGS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

bert_path = '/home/xiaoxf/embedding/bert_base_uncased_en'
root_path = ''

flags.DEFINE_string("gpu_num", '1',
                    "which gpu to use.")

flags.DEFINE_string(
    "output_dir", os.path.join(root_path, 'twitter_325_1epoch_15_newAPI'),
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("train_file", 'twitter/twitter.train1',
                    "train file name")

flags.DEFINE_string("dev_twitter_file", 'twitter/twitter.dev1',
                    "twitter dev file name")

flags.DEFINE_string("dev_reddit_file", 'reddit/reddit.dev1',
                    "reddit dev file name")

flags.DEFINE_string(
    "data_dir", os.path.join(root_path, '/home/xiaoxf/xxf/K-fold/dataset'),
    "The input data dir. ")

flags.DEFINE_string(
    "bert_config_file", os.path.join(bert_path, 'bert_config.json'),
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", os.path.join(bert_path, 'vocab.txt'),
                    "The vocabulary file that the BERT model was trained on.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", os.path.join(bert_path, 'bert_model.ckpt'),
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_eval", True, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", True,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 128, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 2e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 1.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 30000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 100,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")
