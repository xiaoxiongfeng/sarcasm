import tensorflow as tf
import tokenization
import collections
import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from flags import FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_num

class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """

class InputFeatures(object):
    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 # context_ids,
                 # context_mask,
                 # segment1_ids,
                 label_id,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        # self.context_ids = context_ids
        # self.context_mask = context_mask
        # self.segment1_ids = segment1_ids
        self.label_id = label_id
        self.is_real_example = is_real_example

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            # context_ids=[0] * max_seq_length,
            # context_mask=[0] * max_seq_length,
            # segment1_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False)

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)    #tokens_a就是输入，那么问题来了，这里的a和b的区别在哪里呢？
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    # tokens_a = tokenizer.tokenize(example.context_a)
    # tokens_b = None
    # if example.context_b:
    #     tokens_b = tokenizer.tokenize(example.context_b)
    #
    # if tokens_b:
    #     _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    # else:
    #     # Account for [CLS] and [SEP] with "- 2"
    #     if len(tokens_a) > max_seq_length - 2:
    #         tokens_a = tokens_a[0:(max_seq_length - 2)]
    # tokens = []
    # segment1_ids = []
    # tokens.append("[CLS]")
    # segment1_ids.append(0)
    # for token in tokens_a:
    #     tokens.append(token)  # tokens_a就是输入，那么问题来了，这里的a和b的区别在哪里呢？
    #     segment1_ids.append(0)
    # tokens.append("[SEP]")
    # segment1_ids.append(0)
    #
    # if tokens_b:
    #     for token in tokens_b:
    #         tokens.append(token)
    #         segment1_ids.append(1)
    #     tokens.append("[SEP]")
    #     segment1_ids.append(1)
    #
    # context_ids = tokenizer.convert_tokens_to_ids(tokens)
    #
    # # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # # tokens are attended to.
    # context_mask = [1] * len(context_ids)

    # Zero-pad up to the sequence length.
    # while len(context_ids) < max_seq_length:
    #     context_ids.append(0)
    #     context_mask.append(0)
    #     segment1_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    # assert len(context_ids) == max_seq_length
    # assert len(context_mask) == max_seq_length
    # assert len(segment1_ids) == max_seq_length

    labels_ids = []
    for label in example.label:   #tokens_a = tokenizer.tokenize(example.text_a)
        labels_ids.append(int(label))

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: {} (id = {})".format(example.label, labels_ids))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        # context_ids=context_ids,
        # context_mask=context_mask,
        # segment1_ids=segment1_ids,
        label_id=labels_ids,
        is_real_example=True)
    return feature

def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        # features["context_ids"] = create_int_feature(feature.context_ids)
        # features["context_mask"] = create_int_feature(feature.context_mask)
        # features["segment1_ids"] = create_int_feature(feature.segment1_ids)
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])
        if isinstance(feature.label_id, list):
            label_ids = feature.label_id
        else:
            label_ids = [feature.label_id]

        features["label_ids"] = create_int_feature(label_ids)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        features.append(feature)
    return features

def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        # "context_ids": tf.FixedLenFeature([seq_length], tf.int64),
        # "context_mask": tf.FixedLenFeature([seq_length], tf.int64),
        # "segment1_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([2], tf.int64),         #原来在这里写死了它的类别是6啊
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }
    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        #print("examplepppp", example)
        return example

    def input_fn(params):
        """The actual input function."""
        # batch_size = params["batch_size"]
        batch_size = 32

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn