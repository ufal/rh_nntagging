class Vocab(dict):
    """Vocab holds mapping between string tokens and integer ids."""
    def __init__(self):
        super(Vocab, self).__init__()

        self._rev = dict()
        self.frozen = False

    def freeze(self):
        """Freeze the mapping and raise exception if anyone is trying to add values into it."""
        self.frozen = True

    def add(self, word):
        """Add given word to the mapping and return its id."""
        if not word in self:
            if self.frozen:
                raise KeyError

            val = len(self)
            self[word] = val
            self._rev[val] = word

        return self[word]

    def rev(self, word_id):
        """Return the string token for given id."""
        return self._rev[word_id]