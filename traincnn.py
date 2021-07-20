import tensorflow as tf
import numpy as np
import os
import datetime
import time
import pickle as pkl
from lstm import rnn_clf
import data_helpers
import data_helper
import data_helpersmul
from sklearn.model_selection import train_test_split

# Parameters
# ==================================================
tf.flags.DEFINE_string('clf', 'clstm', "Type of classifiers. Default: cnn. You have four choices: [cnn, lstm, blstm, clstm]")

# Data loading params
tf.flags.DEFINE_string("trainIncDir", "BisaPlis", "Path of incident data")
tf.flags.DEFINE_string("inc_dir", "data/rt-polaritydata/kejadianpentingnew", "Path of incident data")
tf.flags.DEFINE_string("non_dir", "data/rt-polaritydata/nonkejadianpenting", "Path of nonincident data")
tf.flags.DEFINE_float("dev_sample_percentage", .2, "Percentage of the training data to use for validation")
tf.flags.DEFINE_integer("max_sentence_length", 200, "Max sentence length in train/test data (Default: 50)")
tf.flags.DEFINE_integer('min_frequency', 0, 'Minimal word frequency')
tf.flags.DEFINE_integer('num_classes', 5, 'Number of classes')
tf.flags.DEFINE_integer('max_length', 0, 'Max document length')
tf.flags.DEFINE_integer('vocab_size', 0, 'Vocabulary size')
tf.flags.DEFINE_float('test_size', 0.1, 'Cross validation test size')

# Model Hyperparameters
tf.flags.DEFINE_string("word2vec", "../rcnn-relation-extraction/wiki.id.new.bin", "Word2vec file with pre-trained embeddings")
tf.flags.DEFINE_integer("word_embedding_dim", 300, "Dimensionality of word embedding (Default: 300)")
tf.flags.DEFINE_integer("context_embedding_dim", 512, "Dimensionality of context embedding(= RNN state size)  (Default: 512)")
tf.flags.DEFINE_integer('embedding_size', 300, 'Word embedding size. For CNN, C-LSTM.')
tf.flags.DEFINE_string('filter_sizes', '3, 4, 5', 'CNN filter sizes. For CNN, C-LSTM.')
tf.flags.DEFINE_integer('num_filters', 128, 'Number of filters per filter size. For CNN, C-LSTM.')
tf.flags.DEFINE_integer("hidden_size", 512, "Size of hidden layer (Default: 512)")
tf.flags.DEFINE_integer('num_layers', 2, 'Number of the LSTM cells. For LSTM, Bi-LSTM, C-LSTM')
tf.flags.DEFINE_float('keep_prob', 0.7, 'Dropout keep probability')  # All
tf.flags.DEFINE_float('l2_reg_lambda', 0.5, 'L2 regularization lambda')  # All

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (Default: 64)")
tf.flags.DEFINE_integer("num_epochs", 15, "Number of training epochs (Default: 10)")
tf.flags.DEFINE_integer("display_every", 20, "Number of iterations to display training info.")
tf.flags.DEFINE_integer("evaluate_every_steps", 20, "Evaluate model on dev set after this many steps")
tf.flags.DEFINE_integer('save_every_steps', 100, 'Save the model after this many steps')
tf.flags.DEFINE_integer("checkpoint_every", 20, "Save model after this many steps")
tf.flags.DEFINE_integer("num_checkpoints", 40, "Number of checkpoints to store")
tf.flags.DEFINE_float("learning_rate", 1e-3, "Which learning rate to start with. (Default: 1e-3)")
tf.flags.DEFINE_float('decay_rate', 1, 'Learning rate decay rate. Range: (0, 1]')  # Learning rate decay
tf.flags.DEFINE_integer('decay_steps', 100000, 'Learning rate decay steps')  # Learning rate decay

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
# Output directory for models and summaries
timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
if not os.path.exists(out_dir):
	os.makedirs(out_dir)
print("Writing to {}\n".format(out_dir))
params = FLAGS.flag_values_dict()
params['hidden_size'] = len(list(map(int, params['filter_sizes'].split(",")))) * params['num_filters']
params_dict = sorted(params.items(), key=lambda x: x[0])
print('Parameters:')
for item in params_dict:
	print('{}: {}'.format(item[0], item[1]))
print('')

# Save parameters to file
params_folder = os.path.join(out_dir, 'params.pkl')
params_file = open(params_folder, 'wb')
pkl.dump(params, params_file, True)
params_file.close()

def train():
	FLAGS.hidden_size = len(FLAGS.filter_sizes.split(",")) * FLAGS.num_filters
	with tf.device('/cpu:0'):
		x_text, y, lengths, vocab_processor = data_helpers.load_data(file_path=FLAGS.trainIncDir,
                                                               sw_path=None,
                                                               min_frequency=FLAGS.min_frequency,
                                                               max_length=FLAGS.max_length,
                                                               shuffle=True)

	FLAGS.vocab_size = len(vocab_processor.vocabulary_._mapping)
	FLAGS.max_length = vocab_processor.max_document_length

    # x = np.array(list(text_vocab_processor.fit_transform(x_text)))
    # print("Text Vocabulary Size: {:d}".format(len(text_vocab_processor.vocabulary_)))

    # FLAGS.vocab_size = len(text_vocab_processor.vocabulary_)
    # print(len(text_vocab_processor.vocabulary_))
    # FLAGS.max_length = text_vocab_processor.max_document_length

    # print("x = {0}".format(x.shape))
    # print("y = {0}".format(y.shape))
    # print("")

    # # Randomly shuffle data
    # np.random.seed(10)
    # shuffle_indices = np.random.permutation(np.arange(len(y)))
    # x_shuffled = x[shuffle_indices]
    # y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    # dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    # x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    # y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    # print("Train/Dev split: {:d}/{:d}\n".format(len(y_train), len(y_dev)))
    # print(len(x_train))
    # tf.flags.DEFINE_integer("shape", x_train.shape[1], "shape sequence_length")

	x_train, x_valid, y_train, y_valid, train_lengths, valid_lengths = train_test_split(x_text,
                                                                                    y,
                                                                                    lengths,
                                                                                    test_size=FLAGS.test_size,
                                                                                    random_state=22)

	train_data = data_helper.batch_iter(x_train, y_train, train_lengths, FLAGS.batch_size, FLAGS.num_epochs)

	with tf.Graph().as_default():
		with tf.Session() as sess:
			classifier = rnn_clf(FLAGS)

            # Train procedure
			global_step = tf.Variable(0, name='global_step', trainable=False)
            # Learning rate decay
			starter_learning_rate = FLAGS.learning_rate
			learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                                    global_step,
                                                    FLAGS.decay_steps,
                                                    FLAGS.decay_rate,
                                                    staircase=True)
			optimizer = tf.train.AdamOptimizer(learning_rate)
			grads_and_vars = optimizer.compute_gradients(classifier.cost)
			train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            # train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(classifier.loss, global_step=global_step)

            # Summaries for loss and accuracy
			loss_summary = tf.summary.scalar('Loss', classifier.cost)
			accuracy_summary = tf.summary.scalar('Accuracy', classifier.accuracy)

			# Train summary
			train_summary_op = tf.summary.merge_all()
			train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
			train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

			# Validation summary
			valid_summary_op = tf.summary.merge_all()
			valid_summary_dir = os.path.join(out_dir, 'summaries', 'valid')
			valid_summary_writer = tf.summary.FileWriter(valid_summary_dir, sess.graph)

			checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
			checkpoint_prefix = os.path.join(checkpoint_dir, "model")
			if not os.path.exists(checkpoint_dir):
				os.makedirs(checkpoint_dir)

			saver = tf.train.Saver(max_to_keep=FLAGS.num_checkpoints)

			vocab_processor.save(os.path.join(out_dir, 'vocab'))


			sess.run(tf.global_variables_initializer())
			
			def run_step(input_data, is_training=True):
				"""Run one step of the training process."""
				input_x, input_y, sequence_length = input_data

				fetches = {'step': global_step,
					   'cost': classifier.cost,
					   'accuracy': classifier.accuracy,
					   'learning_rate': learning_rate}
				feed_dict = {classifier.input_x: input_x,
						 classifier.input_y: input_y}

				fetches['final_state'] = classifier.final_state
				feed_dict[classifier.batch_size] = len(input_x)
				feed_dict[classifier.sequence_length] = sequence_length

				if is_training:
					fetches['train_op'] = train_op
					fetches['summaries'] = train_summary_op
					feed_dict[classifier.keep_prob] = FLAGS.keep_prob
				else:
					fetches['summaries'] = valid_summary_op
					feed_dict[classifier.keep_prob] = 1.0

				vars = sess.run(fetches, feed_dict)
				step = vars['step']
				cost = vars['cost']
				accuracy = vars['accuracy']
				summaries = vars['summaries']

				# Write summaries to file
				if is_training:
					train_summary_writer.add_summary(summaries, step)
				else:
					valid_summary_writer.add_summary(summaries, step)

				time_str = datetime.datetime.now().isoformat()
				print("{}: step: {}, loss: {:g}, accuracy: {:g}".format(time_str, step, cost, accuracy))

				return accuracy


			print('Start training ...')

			for train_input in train_data:
				run_step(train_input, is_training=True)
				current_step = tf.train.global_step(sess, global_step)

				if current_step % FLAGS.evaluate_every_steps == 0:
					print('\nValidation')
					run_step((x_valid, y_valid, valid_lengths), is_training=False)
					print('')

				if current_step % FLAGS.save_every_steps == 0:
					save_path = saver.save(sess, os.path.join(out_dir, 'model/clf'), current_step)

				if current_step % FLAGS.checkpoint_every == 0:
					save_path = saver.save(sess, checkpoint_prefix, current_step)

			print('\nAll the files have been saved to {}\n'.format(out_dir))

def main(_):
	train()


if __name__ == "__main__":
	tf.app.run()
