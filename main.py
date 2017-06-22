import csv
import tensorflow as tf
import input_data

CLASS_NUM   = 8
NODE_NUM    = 128
SEQ_MAX_LEN = 40  #歌名サイズ
INPUT_DIV   = 1   #入力データの次元
HIDDEN      = 128
STEP_NUM    = 20000
BATCH_SIZE  = 64
NUM_LAYER = 2

def main():
  data = input_data.InputData()
  x = tf.placeholder(tf.float32, [None, SEQ_MAX_LEN, INPUT_DIV])
  t = tf.placeholder(tf.int32, [None])
  seqlen = tf.placeholder(tf.int32, [None]) # 可変長の入力データ（曲名）

  t_one_hot = tf.one_hot(t, depth=CLASS_NUM, dtype=tf.float32) 
  def cell():
    return tf.contrib.rnn.BasicRNNCell(num_units=NODE_NUM, activation=tf.nn.tanh) #中間層のセル
  cells = tf.contrib.rnn.MultiRNNCell([cell() for _ in range(NUM_LAYER)])
  outputs, states = tf.nn.dynamic_rnn(cell=cells, inputs=x, dtype=tf.float32, time_major=False)
  outputs = tf.transpose(outputs, perm=[1, 0, 2])

  w = tf.Variable(tf.random_normal([NODE_NUM, CLASS_NUM], stddev=0.01))
  b = tf.Variable(tf.zeros([CLASS_NUM]))
  logits = tf.matmul(outputs[-1], w) + b  # 出力層
  pred = tf.nn.softmax(logits)  # ソフトマックス

  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=t_one_hot, logits=logits)
  loss = tf.reduce_mean(cross_entropy)  # 誤差関数
  train_step = tf.train.AdamOptimizer().minimize(loss)  # 学習アルゴリズム

  correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(t_one_hot,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 精度

  # 学習の実行
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  i = 0
  for _ in range(STEP_NUM):
      sings,labels = data.next_batch()
      
      length_div = [len(sing) for sing in sings]
      _, _loss, _accuracy = sess.run((train_step, loss, accuracy), feed_dict={x: sings, t: labels, seqlen: length_div })
      
      i += 1
      if i % 100 == 0:
          x_test,t_test = data.test_data()
          loss_test_, acc_test_, pred_ = sess.run([loss, accuracy, pred], feed_dict={x:x_test,t:t_test})
          print("[TRAIN] loss : %f, accuracy : %f" %(_loss, _accuracy))
          print("[TEST loss : %f, accuracy : %f" %(loss_test_, acc_test_))
          output_prediction(pred_)
  sess.close()

  
def output_prediction(prediction):
  categories = _categories()
  maxs = [p.argmax(0) for p in prediction]
  result = {k:len(list(filter(lambda x: x == int(v), maxs))) / len(maxs) for k,v in categories.items()}
  print(result)

def _categories():
  with open('../data/categories.csv', 'r') as f:
    reader = csv.reader(f)
    return {row[0]:row[1] for row in reader}


  

if __name__ == '__main__':
  main()