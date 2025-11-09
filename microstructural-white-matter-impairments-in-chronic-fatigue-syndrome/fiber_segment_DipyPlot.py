import numpy as np

from dipy.data import fetch_bundle_atlas_hcp842, get_two_hcp842_bundles
from dipy.io.streamline import load_trk, load_tractogram
from dipy.stats.analysis import assignment_map
from dipy.viz import actor, window

model_af_l_file = '/home/clancy/TemplateFlow/HCP_YA1065_tractography_all_tracks_trk/association/C_FPH_L.trk'
sft_af_l = load_trk(model_af_l_file, reference="same", bbox_valid_check=False)
model_af_l = sft_af_l.streamlines # (151824, 3)

interactive = True

scene = window.Scene()
scene.SetBackground(1, 1, 1)
scene.add(actor.line(model_af_l, fake_tube=True, linewidth=6))
scene.set_camera(
    focal_point=(-18.17281532, -19.55606842, 6.92485857),
    position=(-360.11, -30.46, -40.44),
    view_up=(-0.03, 0.028, 0.89),
)
#window.record(scene=scene, out_path="af_l_before_assignment_maps.png", size=(600, 600))
if interactive:
    window.show(scene)


rng = np.random.default_rng()
n = 50
indx = assignment_map(model_af_l, model_af_l, n) # (151824,)
indx = np.array(indx)

colors = [rng.random(3) for si in range(n)] # 100

disks_color = []
for i in range(len(indx)):
    disks_color.append(tuple(colors[indx[i]]))


scene = window.Scene()
scene.SetBackground(1, 1, 1)
scene.add(actor.line(model_af_l, fake_tube=True, colors=disks_color, linewidth=6)) # model_af_l: (151824, 3); disks_color:151824, each 3
scene.set_camera(
    focal_point=(-18.17281532, -19.55606842, 6.92485857),
    position=(-360.11, -30.46, -40.44),
    view_up=(-0.03, 0.028, 0.89),
)
#window.record(scene=scene, out_path="af_l_after_assignment_maps.png", size=(600, 600))
window.show(scene)
