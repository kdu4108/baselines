import tensorflow as tf
with tf.Session() as sess:
    saver = tf.train.import_meta_graph(‘models/model-atari-prior-duel-breakout-1/saved.meta’)
    saver.restore(sess, "models/model-atari-prior-duel-breakout-1/saved")
