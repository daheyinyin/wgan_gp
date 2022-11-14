
# ============================================================================

from mindspore import ops, nn
import mindspore.numpy as mnp

class GenWithLossCell(nn.Cell):
    """Generator with loss(wrapped)"""

    def __init__(self, netG, netD):
        super(GenWithLossCell, self).__init__()
        self.netG = netG
        self.netD = netD

    def construct(self, noise):

        fake = self.netG(noise)
        errG = self.netD(fake)
        return -errG


class DisWithLossCell(nn.Cell):
    """ Discriminator with loss(wrapped) """

    def __init__(self, netG, netD):
        super(DisWithLossCell, self).__init__()
        self.netG = netG
        self.netD = netD
        self.gradop = ops.GradOperation()
        self.LAMBDA = 10# 100
        self.uniform = ops.UniformReal()

    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""

        # Get random interpolation between real and fake samples
        alpha = self.uniform((real_samples.shape[0], 1, 1, 1))
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples))

        grad_fn = self.gradop(self.netD)
        gradients = grad_fn(interpolates)
        gradients = gradients.view(gradients.shape[0], -1)
        gradient_penalty = ops.reduce_mean(((mnp.norm(gradients, 2, axis=1) - 1) ** 2))
        return gradient_penalty

    def construct(self, real, noise):

        errD_real = self.netD(real)
        fake = self.netG(noise)
        fake = ops.stop_gradient(fake)
        errD_fake = self.netD(fake)

        #gradient_penalty = self.compute_gradient_penalty(real, fake)

        return errD_fake - errD_real #+ gradient_penalty * self.LAMBDA
