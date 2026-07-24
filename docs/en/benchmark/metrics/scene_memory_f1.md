# Scene Memory F1

**scene_memory_f1** measures whether geometry observed before a camera revisit remains geometrically supported after the revisit. Unlike image-pair similarity metrics, it scores **scene memory** from 3D point-cloud overlap between an observation cloud and a revisit cloud. WorldArena implements it through the `point_cloud_f1` backend on long-horizon action suites where the agent leaves and later returns to a previously seen region.

Long-horizon world models must retain stable scene structure across extended rollouts. A model can look visually consistent frame-to-frame while still forgetting walls, furniture, or layout elements once the viewpoint returns. Scene Memory F1 targets that failure mode directly: it asks how much of the earlier observation cloud can still be matched in 3D after revisit, and how much of the revisit cloud is unsupported by the earlier observation.

The runtime first resolves observation and revisit point clouds. When explicit observation/revisit sidecars or a bundled `scene_memory` NPZ archive are available, those clouds are loaded directly. Otherwise the metric reconstructs both clouds from the action payload and ViPE depth sidecars associated with the prediction. Invalid points are removed, and the clouds may be voxel-downsampled when `voxel_size` is configured. If `centroid_align = true` (the default), the revisit cloud is translated so that its centroid matches the observation centroid before scoring. A distance threshold \(\tau_d\) is then chosen from `distance_threshold` when provided, or from `distance_threshold_ratio * scene_diagonal` otherwise, with default ratio `0.02`.

Let \(P_{obs}\) denote the observation point cloud and \(P_{rev}\) the aligned revisit cloud. For a point \(p\) and cloud \(P\), let \(d(p, P)\) be the nearest-neighbor distance from \(p\) to any point in \(P\). **Retention** measures how much of the earlier observation survives at revisit:

\[
R_{mem}
=
\frac{
    |\{p \in P_{obs}: d(p, P_{rev}) < \tau_d\}|
}{
    |P_{obs}|
}
\tag{1}
\]

**Hallucination** measures revisit geometry that is not supported by the earlier observation. It is converted to a precision-like term before F1 aggregation:

\[
R_{hall}
=
\frac{
    |\{q \in P_{rev}: d(q, P_{obs}) \ge \tau_d\}|
}{
    |P_{rev}|
}
\tag{2}
\]

\[
P_{mem} = 1 - R_{hall}
\tag{3}
\]

The final scene-memory score combines retention and precision symmetrically:

\[
S_{scene}
=
\frac{
    2P_{mem}R_{mem}
}{
    P_{mem}+R_{mem}
}
\tag{4}
\]

Higher \(S_{scene}\) indicates stronger scene memory: more of the previously observed geometry is retained, and fewer unsupported revisit points are hallucinated. The metric evaluates geometric persistence across revisits, not photometric similarity, semantic prompt adherence, or absolute metric-scale reconstruction accuracy.
