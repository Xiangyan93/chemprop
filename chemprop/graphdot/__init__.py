from graphdot.kernel.marginalized import MarginalizedGraphKernel
from graphdot.microkernel import (
    TensorProduct,
    SquareExponential as sExp,
    KroneckerDelta as kDelta,
    Convolution as kConv,
)
from graphdot.microprobability import (
    Additive as Additive_p,
    Constant,
    UniformProbability,
    AssignProbability
)
k = 0.90
k_bounds = (0.5, 1.0)
knode = TensorProduct(
    AtomicNumber=kDelta(k, k_bounds),
    AtomicNumber_list_1=kConv(kDelta(k, k_bounds)),
    AtomicNumber_list_2=kConv(kDelta(k, k_bounds)),
    AtomicNumber_list_3=kConv(kDelta(k, k_bounds)),
    AtomicNumber_list_4=kConv(kDelta(k, k_bounds)),
    AtomicNumber_count_1=kDelta(k, k_bounds),
    AtomicNumber_count_2=kDelta(k, k_bounds),
    MorganHash=kDelta(k, k_bounds),
    Ring_count=kDelta(k, k_bounds),
    RingSize_list=kConv(kDelta(k, k_bounds)),
    Hcount=kDelta(k, k_bounds),
    Chiral=kDelta(k, k_bounds),
)
kedge = TensorProduct(
    Order=kDelta(k, k_bounds),
    Stereo=kDelta(k, k_bounds),
    RingStereo=kDelta(k, k_bounds),
    Conjugated=kDelta(k, k_bounds)
)
start_probability=Additive_p(
    AtomicNumber=Constant(1.0)
)
kernel = MarginalizedGraphKernel(
    node_kernel=knode,
    edge_kernel=kedge,
    q=0.01,
    q_bounds=(1e-3, 0.5),
    p=start_probability
)
