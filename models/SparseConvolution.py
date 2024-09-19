from networkUtils import *

class SparseConvolution(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        cs = [int(0.64 * x) for x in cs]

        self.stem = nn.Sequential(
            torchsparse.nn.Conv3d(3, cs[0], kernel_size=3, stride=1),
            torchsparse.nn.BatchNorm(cs[0]), torchsparse.nn.ReLU(True),
            torchsparse.nn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1),
            torchsparse.nn.BatchNorm(cs[0]), torchsparse.nn.ReLU(True))

        self.stage1 = nn.Sequential(
            SparseConv3d(cs[0], cs[0], ks=2, stride=2, dilation=1),
            SparseConv3dRes(cs[0], cs[1], ks=3, stride=1, dilation=1),
            SparseConv3dRes(cs[1], cs[1], ks=3, stride=1, dilation=1),
        )

        self.stage2 = nn.Sequential(
            SparseConv3d(cs[1], cs[1], ks=2, stride=2, dilation=1),
            SparseConv3dRes(cs[1], cs[2], ks=3, stride=1, dilation=1),
            SparseConv3dRes(cs[2], cs[2], ks=3, stride=1, dilation=1),
        )

        self.stage3 = nn.Sequential(
            SparseConv3d(cs[2], cs[2], ks=2, stride=2, dilation=1),
            SparseConv3dRes(cs[2], cs[3], ks=3, stride=1, dilation=1),
            SparseConv3dRes(cs[3], cs[3], ks=3, stride=1, dilation=1),
        )

        self.stage4 = nn.Sequential(
            SparseConv3d(cs[3], cs[3], ks=2, stride=2, dilation=1),
            SparseConv3dRes(cs[3], cs[4], ks=3, stride=1, dilation=1),
            SparseConv3dRes(cs[4], cs[4], ks=3, stride=1, dilation=1),
        )

        self.up1 = nn.ModuleList([
            SparseDeConv3d(cs[4], cs[5], ks=2, stride=2),
            nn.Sequential(
                SparseConv3dRes(cs[5] + cs[3], cs[5], ks=3, stride=1,
                              dilation=1),
                SparseConv3dRes(cs[5], cs[5], ks=3, stride=1, dilation=1),
            )
        ])

        self.up2 = nn.ModuleList([
            SparseDeConv3d(cs[5], cs[6], ks=2, stride=2),
            nn.Sequential(
                SparseConv3dRes(cs[6] + cs[2], cs[6], ks=3, stride=1,
                              dilation=1),
                SparseConv3dRes(cs[6], cs[6], ks=3, stride=1, dilation=1),
            )
        ])

        self.up3 = nn.ModuleList([
            SparseDeConv3d(cs[6], cs[7], ks=2, stride=2),
            nn.Sequential(
                SparseConv3dRes(cs[7] + cs[1], cs[7], ks=3, stride=1,
                              dilation=1),
                SparseConv3dRes(cs[7], cs[7], ks=3, stride=1, dilation=1),
            )
        ])

        self.up4 = nn.ModuleList([
            SparseDeConv3d(cs[7], cs[8], ks=2, stride=2),
            nn.Sequential(
                SparseConv3dRes(cs[8] + cs[0], cs[8], ks=3, stride=1,
                              dilation=1),
                SparseConv3dRes(cs[8], cs[8], ks=3, stride=1, dilation=1),
            )
        ])

        self.classifier = nn.Sequential(nn.Linear(cs[8], 32))

        self.point_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cs[0], cs[4]),
                nn.BatchNorm1d(cs[4]),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[4], cs[6]),
                nn.BatchNorm1d(cs[6]),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[6], cs[8]),
                nn.BatchNorm1d(cs[8]),
                nn.ReLU(True),
            )
        ])

        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.dropout = nn.Dropout(0.3, True)

    def forward(self, x):
        pt = PointTensor(x.F, x.C.float())

        vt = initial_voxelize(pt, 0.05, 0.05)

        vt = self.stem(vt)
        pt0 = voxel_to_point(vt, pt, nearest=False)
        pt0.F = pt0.F
        dsFeature1 = point_to_voxel(vt, pt0)
        dsFeature1 = self.stage1(dsFeature1)
        dsFeature2 = self.stage2(dsFeature1)
        dsFeature3 = self.stage3(dsFeature2)
        dsFeature4 = self.stage4(dsFeature3)
        pt1 = voxel_to_point(dsFeature4, pt0)
        pt1.F = pt1.F + self.point_transforms[0](pt0.F)
        upFeature1 = point_to_voxel(dsFeature4, pt1)
        upFeature1.F = self.dropout(upFeature1.F)
        upFeature1 = self.up1[0](upFeature1)
        upFeature1 = torchsparse.cat([upFeature1, dsFeature3])
        upFeature1 = self.up1[1](upFeature1)
        upFeature2 = self.up2[0](upFeature1)
        upFeature2 = torchsparse.cat([upFeature2, dsFeature2])
        upFeature2 = self.up2[1](upFeature2)
        pt2 = voxel_to_point(upFeature2, pt1)
        pt2.F = pt2.F + self.point_transforms[1](pt1.F)
        upFeature3 = point_to_voxel(upFeature2, pt2)
        upFeature3.F = self.dropout(upFeature3.F)
        upFeature3 = self.up3[0](upFeature3)
        upFeature3 = torchsparse.cat([upFeature3, dsFeature1])
        upFeature3 = self.up3[1](upFeature3)
        upFeature4 = self.up4[0](upFeature3)
        upFeature4 = torchsparse.cat([upFeature4, vt])
        upFeature4 = self.up4[1](upFeature4)
        pt3 = voxel_to_point(upFeature4, pt2)
        pt3.F = pt3.F + self.point_transforms[2](pt2.F)
        out = self.classifier(pt3.F)
        return out


