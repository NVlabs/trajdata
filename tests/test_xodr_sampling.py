import numpy as np

from trajdata.dataset_specific.xodr.parser import parse_xodr


def test_lane_widths_use_actual_samples_for_subresolution_roads():
    xodr = """<?xml version="1.0" standalone="yes"?>
<OpenDRIVE>
  <road name="connector" length="4.47545209131181e-16" id="43" junction="60">
    <planView>
      <geometry s="0" x="3.4109831212174218" y="0.17676254182897455"
                hdg="5.894318153732936" length="4.47545209131181e-16">
        <line/>
      </geometry>
    </planView>
    <lanes>
      <laneSection s="0">
        <center><lane id="0" type="none"/></center>
        <right>
          <lane id="-1" type="driving">
            <width sOffset="0" a="4.283505748483188" b="3.390282745438411"
                   c="-36028797018964280" d="5.366876356306281e31"/>
          </lane>
        </right>
      </laneSection>
    </lanes>
  </road>
</OpenDRIVE>
"""

    parsed = parse_xodr(xodr, resolution=0.5)
    lane = parsed.lanes["43_-1"]

    assert np.isfinite(lane.center).all()
    assert np.abs(lane.center).max() < 10.0
    assert np.linalg.norm(lane.center[1] - lane.center[0]) < 1e-6
