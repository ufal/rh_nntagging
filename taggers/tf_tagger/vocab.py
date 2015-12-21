class Vocab(dict):
    """Vocab holds mapping between string tokens and integer ids."""
    def __init__(self):
        super(Vocab, self).__init__()

        self._rev = {}
        self.frozen = False
        self.counts = []

    def freeze(self):
        """Freeze the mapping and raise exception if anyone is trying to add values into it."""
        self.frozen = True

    def add(self, word):
        """Add given word to the mapping and return its id."""
        if not word in self:
            if self.frozen:
                raise KeyError

            word_id = len(self)
            self[word] = word_id
            self._rev[word_id] = word
            self.counts.append(0)
        else:
            word_id = self[word]

        self.counts[word_id] += 1
        return word_id

    def rev(self, word_id):
        """Return the string token for given id."""
        return self._rev[word_id]

    def count(self, word_id):
        """Returns frequency of word of given id."""
        return self.counts[word_id]
