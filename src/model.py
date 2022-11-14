
# ============================================================================
# create model

from src.dcgan_model import DcganD, DcganG
from src.resgan_model import GoodGenerator, GoodDiscriminator, ResnetGenerator, ResnetDiscriminator



def create_G(model_type, isize, nz, nc, ngf):
    if model_type == 'resnet':
        if isize == 64:
            netG = GoodGenerator(isize, nz, nc, ngf)
        elif isize == 128:
            netG = ResnetGenerator(isize, nz, nc, ngf)
        else:
            raise Exception('invalid resample value')
    elif model_type == 'dcgan':
        netG = DcganG(isize, nz, nc, ngf)
    else:
        raise Exception('invalid resample value')

    return netG

def create_D(model_type, dataset, isize, nc, ngf):
    if model_type == 'resnet':
        if isize == 64:
            netD = GoodDiscriminator(isize, nc, ngf)
        elif isize == 128:
            netD = ResnetDiscriminator(isize, nc, ngf)
        else:
            raise Exception('invalid resample value')
    elif model_type == 'dcgan':
        if dataset == 'lsun':
            netD = DcganD(isize, nc, ngf, normalization_d=True)
        elif dataset == 'cifar10':
            netD = DcganD(isize, nc, ngf, normalization_d=False)
        else:
            raise Exception('invalid resample value')
    else:
        raise Exception('invalid resample value')

    return netD
