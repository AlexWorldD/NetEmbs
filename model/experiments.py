#!/usr/bin/env python
# encoding: utf-8
__author__ = 'Aleksei Maliutin'
"""
experiments.py
Last modified by lex at 2019-02-11.
"""
import tensorflow as tf


class SquareTest(tf.test.TestCase):

    def testSquare(self):
        with self.test_session():
            x = tf.square([2, 3])
            self.assertAllEqual(x.eval(), [4, 9])

if __name__ == '__main__':
    tf.test.main()


