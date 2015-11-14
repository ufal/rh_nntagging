class Vocab(dict):
    def __init__(self):
        super(Vocab, self).__init__()

        self._rev = dict()
        self.frozen = False

    def freeze(self):
        self.frozen = True

    def add(self, word):
        if not word in self:
            if self.frozen:
                raise KeyError

            val = len(self)
            self[word] = val
            self._rev[val] = word

        return self[word]

    def rev(self, word_id):
        return self._rev[word_id]