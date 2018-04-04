"""
Hurray!! this works!!!
            o==+--
            |  |\ \
            |  | \ \    ____________________
            |   \ \ \   |                  |
            |    \ \ \  |  +------------+  |
            |     \ \ \ |  |     (__)   |  |
            |      \ \ \|  |     (oo)   |  |
            |       \ \ |  | o\  .\/.   |  |
            |        \ \|  | | \/    \  |  |
          /---\       \ |  +------------+  |
         /     \       \|                  |
         |     |        |                  |
         \     /        |                  |
          \---/         |                  |
                        |                  |
                     --------------------------
                    (                          )
                     --------------------------
"""

import numpy as np

class CharData:


    def __init__(self, datafile_path, batch_size=40, num_timesteps=40):
        self.batch_size = batch_size
        self.cursor = 0
        self.num_timesteps = num_timesteps

        with open(datafile_path, 'r') as f:
            self.all_text = f.read()
        self.character_set = sorted(list(set(self.all_text)))
        print(len(self.all_text))

    
    def get_vector_for_id(self, size, id):
        temp = np.zeros([size])
        temp[id] = 1
        return temp


    def get_next_batch(self):

        if self.cursor > (len(self.all_text)-self.batch_size):
            self.cursor = 0
            return False, None, None
        self.temp_0_x = []
        self.temp_0_y = []
        for i in range(self.batch_size):
            self.temp_1 = []
            for j in range(self.num_timesteps):
                self.temp_1.append(
                    self.get_vector_for_id(len(self.character_set), 
                        self.character_set.index(
                            self.all_text[self.cursor+i+j]
                        )
                    )
                )
            self.temp_0_x.append(self.temp_1)
            # print(self.cursor+i+j+1)
            self.temp_0_y.append(
                self.get_vector_for_id(len(self.character_set),
                    self.character_set.index(
                        self.all_text[self.cursor+i+j+1]
                    )
                )
            )
        self.cursor += self.batch_size
        return True, self.temp_0_x, self.temp_0_y

    def get_random_seed(self, length):
        start = np.random.randint(len(self.all_text) - length)
        return self.all_text[start:start+length]