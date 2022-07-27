from predify.modules import PCoderN
from predify.networks import PNetSeparateHP
from torch.nn import Sequential, ReLU, ConvTranspose2d, Upsample

class PBranchedNetwork_Conv3SeparateHP(PNetSeparateHP):
    def __init__(self, backbone, build_graph=False, random_init=False, ff_multiplier=(0.3), fb_multiplier=(0.3), er_multiplier=(0.01)):
        super().__init__(backbone, 1, build_graph, random_init, ff_multiplier, fb_multiplier, er_multiplier)

        # PCoder number 1
        pmodule = Sequential(Upsample(scale_factor=(23.428571428571427, 23.529411764705884), mode='bilinear', align_corners=False),ConvTranspose2d(512, 1, kernel_size=3, stride=1, padding=1))
        self.pcoder1 = PCoderN(pmodule, False, self.random_init)
        def fw_hook1(m, m_in, m_out):
            e = self.pcoder1(ff=m_out, fb=None, target=self.input_mem, build_graph=self.build_graph, ffm=self.ffm1, fbm=self.fbm1, erm=self.erm1)
            return e[0]
        self.backbone.speech_branch.conv3.block[0].register_forward_hook(fw_hook1)


