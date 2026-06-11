import numpy as np, pytest
from cerebrum.workspace import Workspace

def test_one_hot_write_takes_winner_content():
    ws = Workspace(k_slots=2, content_dim=3)
    z = np.array([[1.0,0.0],[0.0,1.0],[0.0,0.0]])      # module0 -> slot0, module1 -> slot1
    reads = np.array([[1.,1.,1.],[2.,2.,2.],[9.,9.,9.]])
    ws.write(z, reads)
    assert np.allclose(ws.slots[0], [1,1,1]) and np.allclose(ws.slots[1], [2,2,2])

def test_soft_weights_are_rejected():
    ws = Workspace(k_slots=1, content_dim=2)
    zsoft = np.array([[0.6],[0.4]])                    # soft mixing weights = BAN-1
    with pytest.raises(AssertionError):
        ws.write(zsoft, np.array([[1.,0.],[0.,1.]]))

def test_broadcast_sums_slot_contents():
    ws = Workspace(k_slots=2, content_dim=2)
    ws.slots[0] = [1.,0.]; ws.slots[1] = [0.,3.]
    assert np.allclose(ws.broadcast(), [1.,3.])
