# coding=utf-8
import os
import modeling
import tokenization
import tensorflow as tf

#to util
from util import MultiLabelTextProcessor
from examples import file_based_convert_examples_to_features, file_based_input_fn_builder
from flags import FLAGS
from model import model_fn_builder

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_num

def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case, FLAGS.init_checkpoint)
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    tf.gfile.MakeDirs(FLAGS.output_dir)
    processor = MultiLabelTextProcessor()
    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    # construct estimator
    tf.logging.info("load estimator ...")
    config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=True, gpu_options={"allow_growth": True})
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.99

    run_config = tf.estimator.RunConfig(
        session_config=config,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps)
    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)

        num_train_steps = int(len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn, config=run_config
    )

    print("train")
    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    print("eval")
    def model_dev(eval_examples, type_data):
        eval_file = os.path.join(FLAGS.output_dir, "eval_" + type_data + ".tf_record")
        file_based_convert_examples_to_features(
            eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

        eval_steps = None

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results_" + type_data + ".txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("%s = %s\n" % (key, str(result[key])))
    if FLAGS.do_eval:
        eval_examples = processor.get_dev_twitter_examples(FLAGS.data_dir)
        model_dev(eval_examples, "twitter")
        eval_examples = processor.get_dev_reddit_examples(FLAGS.data_dir)
        model_dev(eval_examples, "reddit")

    print("predict")
    def model_test(predict_examples, type_data):  # type_data = 'reddit'
        num_actual_predict_examples = len(predict_examples)
        predict_file = os.path.join(FLAGS.output_dir, "predict_" + type_data + ".tf_record")
        file_based_convert_examples_to_features(predict_examples, label_list,
                                                FLAGS.max_seq_length, tokenizer,
                                                predict_file)

        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)
        output_predict_file = os.path.join(FLAGS.output_dir, "my_" + type_data + "_answer.tsv")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            num_written_lines = 0
            i = 0
            for predict_example, prediction in zip(predict_examples, result):
                my_label = prediction["my_label"]
                if i >= num_actual_predict_examples:
                    break
                labels = processor.get_labels()
                writer.write(type_data + "_" + predict_examples[i].guid + ',' + labels[my_label] + "\n")
                num_written_lines += 1
                i += 1
    if FLAGS.do_predict:
        predict_examples = processor.get_twitter_examples(FLAGS.data_dir)
        model_test(predict_examples, 'twitter')
        predict_examples = processor.get_reddit_examples(FLAGS.data_dir)
        model_test(predict_examples, 'reddit')

if __name__ == "__main__":
    main()