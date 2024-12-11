import unittest
from speechain.utilbox.humanfriendly import (
    round_number,
    pluralize,
    pluralize_raw,
    format_size,
)


class TestHumanFriendly(unittest.TestCase):
    def test_round_number(self):
        self.assertEqual(round_number(1), "1")
        self.assertEqual(round_number(1.0), "1")
        self.assertEqual(round_number(1.234), "1.23")
        self.assertEqual(round_number(1.567), "1.57")
        self.assertEqual(round_number(1.999), "2")
        self.assertEqual(round_number(1.001), "1")
        self.assertEqual(round_number(1.00, keep_width=True), "1.00")

    def test_pluralize_raw(self):
        self.assertEqual(pluralize_raw(1, "test"), "test")
        self.assertEqual(pluralize_raw(2, "test"), "tests")
        self.assertEqual(pluralize_raw(0, "test"), "tests")
        self.assertEqual(pluralize_raw(1.0, "test"), "test")
        self.assertEqual(pluralize_raw(1, "box", "boxes"), "box")
        self.assertEqual(pluralize_raw(2, "box", "boxes"), "boxes")

    def test_pluralize(self):
        self.assertEqual(pluralize(1, "test"), "1 test")
        self.assertEqual(pluralize(2, "test"), "2 tests")
        self.assertEqual(pluralize(0, "test"), "0 tests")
        self.assertEqual(pluralize(1, "box", "boxes"), "1 box")
        self.assertEqual(pluralize(2, "box", "boxes"), "2 boxes")

    def test_format_size(self):
        self.assertEqual(format_size(0), "0 bytes")
        self.assertEqual(format_size(1), "1 byte")
        self.assertEqual(format_size(2), "2 bytes")
        self.assertEqual(format_size(1000), "1 KB")
        self.assertEqual(format_size(1024, binary=True), "1 KiB")
        self.assertEqual(format_size(1000000), "1 MB")
        self.assertEqual(format_size(1024 * 1024, binary=True), "1 MiB")
        self.assertEqual(format_size(1500), "1.5 KB")
        self.assertEqual(format_size(1500, keep_width=True), "1.50 KB")
        self.assertEqual(format_size(1500, binary=True), "1.46 KiB")
        self.assertEqual(format_size(1000**4), "1 TB")
        self.assertEqual(format_size(1024**4, binary=True), "1 TiB")
