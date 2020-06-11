import modeling
import tensorflow as tf
import optimization
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from flags import FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_num

def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)
    output_layer = model.get_pooled_output()

    #the context
    # model1 = modeling.BertModel(
    #     config=bert_config,
    #     is_training=is_training,
    #     input_ids=context_ids,
    #     input_mask=context_mask,
    #     token_type_ids=segment1_ids,
    #     use_one_hot_embeddings=use_one_hot_embeddings)
    # output_layer1 = model1.get_pooled_output()
    #
    # output_layer_con = tf.concat([output_layer, output_layer1], 1)

    hidden_size = output_layer.shape[-1].value
    # hidden_size = output_layer.shape[-1].value + output_layer1.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights1", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)   #dont use dropout

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)      #在这里output_weights是需要训练的啦
        # logits = tf.nn.bias_add(logits, output_bias)
        # logits = logits * output_layer1
        # logits = tf.matmul(logits, output_weights1, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)

        assert isinstance(logits, object)
        probabilities = tf.nn.sigmoid(logits)

        my_label = tf.argmax(probabilities, axis=1)
        labels = tf.cast(labels, tf.float32)
        # print(243567, labels, logits, output_layer)
        per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tf.reduce_mean(per_example_loss)

        return (loss, per_example_loss, logits, probabilities, my_label)

def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, mode):
        # for name in sorted(features.keys()):
        #     tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        # context_ids = features["context_ids"]
        # context_mask = features["context_mask"]
        # segment1_ids = features["segment1_ids"]
        label_ids = features["label_ids"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        #zai这里labels是6
        (total_loss, per_example_loss, logits, probabilities, my_label) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)
        #在这里得到的probabilities也就是最后的分类softmax的结果
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            # tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
            #                 init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, probabilities, is_real_example):

                logits_split = tf.split(probabilities, num_labels, axis=-1)
                label_ids_split = tf.split(label_ids, num_labels, axis=-1)
                # metrics change to auc of every class
                eval_dict = {}
                for j, logits in enumerate(logits_split):
                    label_id_ = tf.cast(label_ids_split[j], dtype=tf.int32)
                    current_auc, update_op_auc = tf.metrics.auc(label_id_, logits)
                    eval_dict[str(j)] = (current_auc, update_op_auc)
                eval_dict['eval_loss'] = tf.metrics.mean(values=per_example_loss)
                return eval_dict

            eval_metrics = metric_fn(per_example_loss, label_ids, probabilities, is_real_example)
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metric_ops=eval_metrics
            )
        else:
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions={"probabilities": probabilities, "my_label":my_label})
        return output_spec

    return model_fn