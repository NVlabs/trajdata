import numpy as np

from trajdata.dataset_specific.xodr.parser import parse_xodr


def test_artificial_road_edges_preserve_road_elevation():
    xodr = """<?xml version="1.0" standalone="yes"?>
<OpenDRIVE>
  <road name="r1" length="10.0" id="1" junction="-1">
    <planView>
      <geometry s="0.0" x="0.0" y="0.0" hdg="0.0" length="10.0"><line/></geometry>
    </planView>
    <elevationProfile>
      <elevation s="0.0" a="76.0" b="0.5" c="0.0" d="0.0"/>
    </elevationProfile>
    <lanes>
      <laneSection s="0.0">
        <center><lane id="0" type="none" level="false"/></center>
        <right>
          <lane id="-1" type="driving" level="false">
            <link/>
            <width sOffset="0.0" a="3.5" b="0" c="0" d="0"/>
          </lane>
        </right>
      </laneSection>
    </lanes>
  </road>
</OpenDRIVE>"""

    parsed = parse_xodr(xodr, resolution=2.5)

    real_right_edge = parsed.road_edges["1_R"]
    artificial_left_edge = parsed.road_edges["1_L"]

    assert np.allclose(artificial_left_edge[:, 2], real_right_edge[:, 2])
    assert not np.allclose(artificial_left_edge[:, 2], 0.0)
