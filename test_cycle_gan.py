from unittest import TestCase
from cycle_gan import CycleGAN

class TestCycleGAN(TestCase):

    def setUp(self) -> None:
        self.gan = CycleGAN(epochs=500, color_depth=1, progrssive=True)

    def test_run(self):
        self.gan.run(debug=False)
