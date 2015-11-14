from unittest import TestCase

from vocab import Vocab


class TestVocab(TestCase):
    def test_add(self):
        v = Vocab()
        v.add('a')
        v.add('b')
        v.add('c')
        v.add('d')
        v.add('a')

        self.assertEqual(len(v), 4)
        self.assertEqual(v['a'], 0)

        self.assertEqual(v.rev(3), 'd')
        self.assertEqual(v.rev(0), 'a')
