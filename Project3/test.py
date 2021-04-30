import os 
import unittest
from FaceCluster import * 


def base_path():
    name=os.path.basename("data/faceCluster_5" )
    print(name)




class TestStringMethods(unittest.TestCase):
 
    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

    def test_k(self):
        self.assertEqual(extract_k("data/faceCluster_5" ),5)

if __name__ == '__main__':
    unittest.main()


