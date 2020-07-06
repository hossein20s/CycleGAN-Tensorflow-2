from unittest import TestCase
from cycle_gan import CycleGAN

class TestCycleGAN(TestCase):

    def setUp(self) -> None:
        self.gan = CycleGAN(epochs=3, color_depth=1)

    def test_run(self):
        self.gan.run(debug=True)
