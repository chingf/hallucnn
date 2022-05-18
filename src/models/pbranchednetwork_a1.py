from predify.modules import PCoderN
from predify.networks import PNetSeparateHP
from torch.nn import Sequential, ReLU, ConvTranspose2d, Upsample

class PBranchedNetwork_A1SeparateHP(PNetSeparateHP):
    def __init__(self, backbone, build_graph=False, random_init=False, ff_multiplier=(0.3,0.3,0.3), fb_multiplier=(0.3,0.3,0.3), er_multiplier=(0.01,0.01,0.01)):
        super().__init__(backbone, 3, build_graph, random_init, ff_multiplier, fb_multiplier, er_multiplier)

        # PCoder number 1
        pmodule = Sequential(Upsample(scale_factor=(2.981818181818182, 2.985074626865672), mode='bilinear', align_corners=False),ConvTranspose2d(96, 1, kernel_size=3, stride=1, padding=1))
        self.pcoder1 = PCoderN(pmodule, True, self.random_init)
        def fw_hook1(m, m_in, m_out):
            e = self.pcoder1(ff=m_out, fb=self.pcoder2.prd, target=self.input_mem, build_graph=self.build_graph, ffm=self.ffm1, fbm=self.fbm1, erm=self.erm1)
            return e[0]
        self.backbone.speech_branch.conv1.block[0].register_forward_hook(fw_hook1)

        # PCoder number 2
        pmodule = Sequential(Upsample(scale_factor=(3.9285714285714284, 3.9411764705882355), mode='bilinear', align_corners=False),ConvTranspose2d(256, 96, kernel_size=3, stride=1, padding=1))
        self.pcoder2 = PCoderN(pmodule, True, self.random_init)
        def fw_hook2(m, m_in, m_out):
            e = self.pcoder2(ff=m_out, fb=self.pcoder3.prd, target=self.pcoder1.rep, build_graph=self.build_graph, ffm=self.ffm2, fbm=self.fbm2, erm=self.erm2)
            return e[0]
        self.backbone.speech_branch.conv2.block[0].register_forward_hook(fw_hook2)

        # PCoder number 3
        pmodule = Sequential(Upsample(scale_factor=(2.0, 2.0), mode='bilinear', align_corners=False),ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1))
        self.pcoder3 = PCoderN(pmodule, False, self.random_init)
        def fw_hook3(m, m_in, m_out):
            e = self.pcoder3(ff=m_out, fb=None, target=self.pcoder2.rep, build_graph=self.build_graph, ffm=self.ffm3, fbm=self.fbm3, erm=self.erm3)
            return e[0]
        self.backbone.speech_branch.conv3.block[0].register_forward_hook(fw_hook3)


