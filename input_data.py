import csv
import random


class InputData(object):

  def __init__(self, data_dir='../data/', batch_size=64, max_length=40):
    self.max_length = max_length
    self.train_data = self.__create_batchs(data_dir)
    self.__test_data, self.__test_target = self.__create_test_data(data_dir)
    self.batch_size = batch_size
    self.idx = 0
    
  
  #バッチサイズ分のデータ取得
  def next_batch(self):
    if self.idx + self.batch_size > len(self.train_data):
      self.idx = 0
      random.shuffle(self.train_data)
    batch = self.train_data[self.idx:self.idx+self.batch_size]
    actuals = [actual for (train, actual) in batch]
    trains = [train for (train, actual) in batch]
    self.idx += self.batch_size
    trains_id = self.data2id(trains)
    return trains_id, actuals

  
  def data2id(self, data):
    self.__create_dict()
    data = [train.lower().replace(' ', '') for train in data]
    return [[[self.char_dict[train[i]]] if len(train) > i else [0] for i in range(self.max_length)] for train in data]
  

  def __create_dict(self, data_dir ='../data/'):
    data = self.__create_batchs(data_dir)
    data += self.__create_batchs(data_dir, test=True)
    sings = [d[0] for d in data]
    word = ''.join(sings).lower().replace(' ','')
    word_uniq = list(set(word))
    self.char_dict = {k:i for i,k in enumerate(word_uniq)}
  
  
  def __create_batchs(self, data_dir, test=False):
    if not test:
      names, genres = self.__read_data(data_dir + 'train_data.csv')
    else:
      names, genres = self.__read_data(data_dir + 'test_data.csv')
    return [(name, int(genre)) for name, genre in zip(names, genres)]
  
  
  def __create_test_data(self, data_dir):
    test_data = self.__create_batchs(data_dir, test=True)
    actuals = [actual for (train, actual) in test_data]
    _input = [train for (train, actual) in test_data]
    return self.data2id(_input), actuals


  def __read_data(self, data_path):
    names = []
    genres = []
    for name, genre in self.__read_one_data(data_path):
      names.append(name)
      genres.append(genre)
    return names, genres

  def __read_one_data(self, data_path):
    with open(data_path, 'r') as f:
      reader = csv.reader(f)
      for row in reader:
        yield row[0], row[1]
  

  def test_data(self):
    return self.__test_data, self.__test_target

IN = InputData()
sings,labels = IN.next_batch()
#print(sings,labels)