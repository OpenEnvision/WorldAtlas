# Physics Analytics

<div class="metric-grid">
<div class="metric-card">
  <div class="metric-label">Physics Rows</div>
  <div class="metric-value">1,400</div>
  <div class="metric-note">All scanned physics videos</div>
</div>
<div class="metric-card">
  <div class="metric-label">Physics Metadata</div>
  <div class="metric-value">377</div>
  <div class="metric-note">Rows matched to the curated subset CSV</div>
</div>
<div class="metric-card">
  <div class="metric-label">Physics Groups</div>
  <div class="metric-value">2</div>
  <div class="metric-note">Top-level physics families</div>
</div>
<div class="metric-card">
  <div class="metric-label">Physics Dimensions</div>
  <div class="metric-value">28</div>
  <div class="metric-note">Fine-grained physics labels</div>
</div>
</div>

## Physics Dimension Summary

| physics_group             | physics_dimension              |   items |   metadata |   avg_duration_seconds |   avg_visual_quality |   avg_motion_score |
|:--------------------------|:-------------------------------|--------:|-----------:|-----------------------:|---------------------:|-------------------:|
| basic_physics_consistency | acceleration                   |      50 |         12 |                3.50877 |              5.26167 |          0.6475    |
| basic_physics_consistency | circular_motion                |      50 |         10 |                4.5841  |              5.304   |          0.786     |
| basic_physics_consistency | collision                      |      50 |         16 |                4.49878 |              5.54875 |          1.01062   |
| basic_physics_consistency | damped_oscillation             |      50 |         10 |                4.74264 |              4.976   |          0.491     |
| basic_physics_consistency | deceleration                   |      50 |          3 |                4.18964 |              4.80333 |          2.49667   |
| basic_physics_consistency | deformation                    |      50 |         18 |                4.99894 |              5.37833 |          1.93833   |
| basic_physics_consistency | elastic_motion                 |      50 |         12 |                5.72252 |              5.73583 |          0.328333  |
| basic_physics_consistency | motion_3d                      |      50 |         12 |                5.53159 |              5.12083 |          6.29333   |
| basic_physics_consistency | parabolic_motion               |      50 |         14 |                4.9173  |              5.08786 |          2.35071   |
| basic_physics_consistency | parabolic_motion_with_rotation |      50 |          6 |                4.61854 |              5.45    |          0.348333  |
| basic_physics_consistency | rigid_body_motion              |      50 |         16 |                5.96114 |              5.5025  |          1.9625    |
| basic_physics_consistency | rotation                       |      50 |         16 |                5.04031 |              5.21438 |          0.51375   |
| basic_physics_consistency | size_changing                  |      50 |         10 |                5.12466 |              4.856   |          0.356     |
| basic_physics_consistency | slope_sliding                  |      50 |          8 |                4.66202 |              4.9     |          5.04375   |
| basic_physics_consistency | uniform_motion                 |      50 |         12 |                5.00959 |              5.42833 |          4.3425    |
| fluid_optics_thermal      | colored_lighting_bloom         |      50 |         16 |                5.85976 |              6.1725  |          3.31312   |
| fluid_optics_thermal      | combustion                     |      50 |         16 |                5.06591 |              5.60062 |          1.69625   |
| fluid_optics_thermal      | directional_lighting           |      50 |         14 |                5.28619 |              5.33929 |          0.985     |
| fluid_optics_thermal      | liquefaction                   |      50 |         10 |                5.2525  |              5.99    |          0.329     |
| fluid_optics_thermal      | liquid_motion                  |      50 |         18 |                6.34561 |              5.72944 |          0.701667  |
| fluid_optics_thermal      | melting                        |      50 |         12 |                6.7429  |              5.30583 |          0.0483333 |
| fluid_optics_thermal      | point_lighting                 |      50 |         14 |                5.16664 |              5.505   |          0.254286  |
| fluid_optics_thermal      | reflection                     |      50 |         24 |                5.98938 |              6.32917 |          0.372083  |
| fluid_optics_thermal      | refraction                     |      50 |         16 |                6.00517 |              5.53688 |          0.65625   |
| fluid_optics_thermal      | scattering                     |      50 |         16 |                6.58354 |              6.06062 |          0.25625   |
| fluid_optics_thermal      | smoke_effects                  |      50 |         20 |                6.30163 |              5.8635  |          0.2975    |
| fluid_optics_thermal      | solidification                 |      50 |         12 |                5.96877 |              5.29167 |          0.7       |
| fluid_optics_thermal      | vaporization                   |      50 |         14 |                5.65296 |              5.49857 |          0.850714  |

## Quality And Motion Summary

| physics_group             |   items |   avg_visual_quality |   avg_motion_score |   avg_selection_score |
|:--------------------------|--------:|---------------------:|-------------------:|----------------------:|
| fluid_optics_thermal      |     202 |              5.75297 |           0.810842 |               12.3861 |
| basic_physics_consistency |     175 |              5.28297 |           1.88109  |               11.3146 |

## Sample Physics Gallery

<div class="sample-grid sample-grid--physics">
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../generated/samples/posters/01-physics_videos-00c09d36eacc9b6bc61a368fff295cd06af132b25e4ed4613ed984b1c81ff31c.jpg"><source src="../generated/samples/previews/01-physics_videos-00c09d36eacc9b6bc61a368fff295cd06af132b25e4ed4613ed984b1c81ff31c.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Basic Physics Consistency / Acceleration</strong>
    <span>videos_by_category/basic_physics_consistency/acceleration/00c09d36eacc9b6bc61a368fff295cd06af132b25e4ed4613ed984b1c81ff31c.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../generated/samples/posters/02-physics_videos-099a31d25b7b9f27d8dca0d86c286737761571d8c20aa8f45d81c1ce1013c162.jpg"><source src="../generated/samples/previews/02-physics_videos-099a31d25b7b9f27d8dca0d86c286737761571d8c20aa8f45d81c1ce1013c162.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Basic Physics Consistency / Circular Motion</strong>
    <span>videos_by_category/basic_physics_consistency/circular_motion/099a31d25b7b9f27d8dca0d86c286737761571d8c20aa8f45d81c1ce1013c162.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../generated/samples/posters/03-physics_videos-00e24098bbcee584cb7e8d5d200186b5dfe38b978ec8cfff842f417c520cc967.jpg"><source src="../generated/samples/previews/03-physics_videos-00e24098bbcee584cb7e8d5d200186b5dfe38b978ec8cfff842f417c520cc967.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Basic Physics Consistency / Collision</strong>
    <span>videos_by_category/basic_physics_consistency/collision/00e24098bbcee584cb7e8d5d200186b5dfe38b978ec8cfff842f417c520cc967.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../generated/samples/posters/04-physics_videos-01891e77e3e9afc366ec41a71d063409c988c3a4a572232e0a3cd3425de31281.jpg"><source src="../generated/samples/previews/04-physics_videos-01891e77e3e9afc366ec41a71d063409c988c3a4a572232e0a3cd3425de31281.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Basic Physics Consistency / Damped Oscillation</strong>
    <span>videos_by_category/basic_physics_consistency/damped_oscillation/01891e77e3e9afc366ec41a71d063409c988c3a4a572232e0a3cd3425de31281.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../generated/samples/posters/05-physics_videos-019f9c756bb4c7260ad87e2d38469dc5b99287bfdf24438c1975a8aad97771ab.jpg"><source src="../generated/samples/previews/05-physics_videos-019f9c756bb4c7260ad87e2d38469dc5b99287bfdf24438c1975a8aad97771ab.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Basic Physics Consistency / Deceleration</strong>
    <span>videos_by_category/basic_physics_consistency/deceleration/019f9c756bb4c7260ad87e2d38469dc5b99287bfdf24438c1975a8aad97771ab.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../generated/samples/posters/06-physics_videos-0185106d958eafa893e759d16434ed1ce35482526e5edd7d94d6f01fe2b4c988.jpg"><source src="../generated/samples/previews/06-physics_videos-0185106d958eafa893e759d16434ed1ce35482526e5edd7d94d6f01fe2b4c988.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Basic Physics Consistency / Deformation</strong>
    <span>videos_by_category/basic_physics_consistency/deformation/0185106d958eafa893e759d16434ed1ce35482526e5edd7d94d6f01fe2b4c988.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../generated/samples/posters/07-physics_videos-0a30d1aa96460386a859a803ec80d60901dd453be737844250cb9c134d7119e2.jpg"><source src="../generated/samples/previews/07-physics_videos-0a30d1aa96460386a859a803ec80d60901dd453be737844250cb9c134d7119e2.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Basic Physics Consistency / Elastic Motion</strong>
    <span>videos_by_category/basic_physics_consistency/elastic_motion/0a30d1aa96460386a859a803ec80d60901dd453be737844250cb9c134d7119e2.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../generated/samples/posters/08-physics_videos-082e01d1b61f9360305144d49feea870487197c2560475116c79b8425bdde234.jpg"><source src="../generated/samples/previews/08-physics_videos-082e01d1b61f9360305144d49feea870487197c2560475116c79b8425bdde234.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Basic Physics Consistency / Motion 3D</strong>
    <span>videos_by_category/basic_physics_consistency/motion_3d/082e01d1b61f9360305144d49feea870487197c2560475116c79b8425bdde234.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../generated/samples/posters/09-physics_videos-019c2446b47f1125aaedced3fb5cae96a66e8bd476f3f34642821a8025d569f1.jpg"><source src="../generated/samples/previews/09-physics_videos-019c2446b47f1125aaedced3fb5cae96a66e8bd476f3f34642821a8025d569f1.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Basic Physics Consistency / Parabolic Motion</strong>
    <span>videos_by_category/basic_physics_consistency/parabolic_motion/019c2446b47f1125aaedced3fb5cae96a66e8bd476f3f34642821a8025d569f1.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../generated/samples/posters/10-physics_videos-082638c19cc7152742981f17dae9c4c3e8e49f45a4bee420a183ca2f3bece4b1.jpg"><source src="../generated/samples/previews/10-physics_videos-082638c19cc7152742981f17dae9c4c3e8e49f45a4bee420a183ca2f3bece4b1.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Basic Physics Consistency / Parabolic Motion With Rotation</strong>
    <span>videos_by_category/basic_physics_consistency/parabolic_motion_with_rotation/082638c19cc7152742981f17dae9c4c3e8e49f45a4bee420a183ca2f3bece4b1.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../generated/samples/posters/11-physics_videos-014999622d62d4e3b15b2c4510ea4e96bb998c9b09f353fa323062675e51e0f4.jpg"><source src="../generated/samples/previews/11-physics_videos-014999622d62d4e3b15b2c4510ea4e96bb998c9b09f353fa323062675e51e0f4.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Basic Physics Consistency / Rigid Body Motion</strong>
    <span>videos_by_category/basic_physics_consistency/rigid_body_motion/014999622d62d4e3b15b2c4510ea4e96bb998c9b09f353fa323062675e51e0f4.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../generated/samples/posters/12-physics_videos-00016ac82302e0bff8dd5cb51c5de66402f311e4f0338e1cc7a60dfd5845f3fe.jpg"><source src="../generated/samples/previews/12-physics_videos-00016ac82302e0bff8dd5cb51c5de66402f311e4f0338e1cc7a60dfd5845f3fe.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Basic Physics Consistency / Rotation</strong>
    <span>videos_by_category/basic_physics_consistency/rotation/00016ac82302e0bff8dd5cb51c5de66402f311e4f0338e1cc7a60dfd5845f3fe.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../generated/samples/posters/13-physics_videos-076b36d51935adaaca1370bc800623bef3ed4faf52d07758c67da038c1a400ad.jpg"><source src="../generated/samples/previews/13-physics_videos-076b36d51935adaaca1370bc800623bef3ed4faf52d07758c67da038c1a400ad.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Basic Physics Consistency / Size Changing</strong>
    <span>videos_by_category/basic_physics_consistency/size_changing/076b36d51935adaaca1370bc800623bef3ed4faf52d07758c67da038c1a400ad.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../generated/samples/posters/14-physics_videos-0a5c32ef7c9a845fa429466c7e2e43225f2de91fab94632ad7e14e31e25f0571.jpg"><source src="../generated/samples/previews/14-physics_videos-0a5c32ef7c9a845fa429466c7e2e43225f2de91fab94632ad7e14e31e25f0571.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Basic Physics Consistency / Slope Sliding</strong>
    <span>videos_by_category/basic_physics_consistency/slope_sliding/0a5c32ef7c9a845fa429466c7e2e43225f2de91fab94632ad7e14e31e25f0571.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../generated/samples/posters/15-physics_videos-06fc5a6aabdcb34a7f9d111775787a3d724b53e086081aae2fd6b6dbc5ce09e8.jpg"><source src="../generated/samples/previews/15-physics_videos-06fc5a6aabdcb34a7f9d111775787a3d724b53e086081aae2fd6b6dbc5ce09e8.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Basic Physics Consistency / Uniform Motion</strong>
    <span>videos_by_category/basic_physics_consistency/uniform_motion/06fc5a6aabdcb34a7f9d111775787a3d724b53e086081aae2fd6b6dbc5ce09e8.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../generated/samples/posters/16-physics_videos-02d4a47269e3ac53142386e4b1f51859471ca5af7f25125800b18bbff2254f71.jpg"><source src="../generated/samples/previews/16-physics_videos-02d4a47269e3ac53142386e4b1f51859471ca5af7f25125800b18bbff2254f71.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Fluid Optics Thermal / Colored Lighting Bloom</strong>
    <span>videos_by_category/fluid_optics_thermal/colored_lighting_bloom/02d4a47269e3ac53142386e4b1f51859471ca5af7f25125800b18bbff2254f71.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../generated/samples/posters/17-physics_videos-05599f730225050093f0d413a692afd99b34a6f722046abb4eb3102d56de753f.jpg"><source src="../generated/samples/previews/17-physics_videos-05599f730225050093f0d413a692afd99b34a6f722046abb4eb3102d56de753f.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Fluid Optics Thermal / Combustion</strong>
    <span>videos_by_category/fluid_optics_thermal/combustion/05599f730225050093f0d413a692afd99b34a6f722046abb4eb3102d56de753f.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../generated/samples/posters/18-physics_videos-00de7a3ef7259ad83c0a221c06db087baa2283cb333df63a2a5c947b17ec5b13.jpg"><source src="../generated/samples/previews/18-physics_videos-00de7a3ef7259ad83c0a221c06db087baa2283cb333df63a2a5c947b17ec5b13.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Fluid Optics Thermal / Directional Lighting</strong>
    <span>videos_by_category/fluid_optics_thermal/directional_lighting/00de7a3ef7259ad83c0a221c06db087baa2283cb333df63a2a5c947b17ec5b13.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../generated/samples/posters/19-physics_videos-0350a292fea1ac777ec4f576040a9b03e0ea9d8fe4c64fb11489d07238b50be1.jpg"><source src="../generated/samples/previews/19-physics_videos-0350a292fea1ac777ec4f576040a9b03e0ea9d8fe4c64fb11489d07238b50be1.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Fluid Optics Thermal / Liquefaction</strong>
    <span>videos_by_category/fluid_optics_thermal/liquefaction/0350a292fea1ac777ec4f576040a9b03e0ea9d8fe4c64fb11489d07238b50be1.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../generated/samples/posters/20-physics_videos-0412e16a6083996bac37b18dbb0e721792f3558bd8a35cbe9a3c2a26ea663c0c.jpg"><source src="../generated/samples/previews/20-physics_videos-0412e16a6083996bac37b18dbb0e721792f3558bd8a35cbe9a3c2a26ea663c0c.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Fluid Optics Thermal / Liquid Motion</strong>
    <span>videos_by_category/fluid_optics_thermal/liquid_motion/0412e16a6083996bac37b18dbb0e721792f3558bd8a35cbe9a3c2a26ea663c0c.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../generated/samples/posters/21-physics_videos-0aeb93ecd62f067df76b9d3f3667d9a19c8a2ff667f3663a26fa86e484a08d86.jpg"><source src="../generated/samples/previews/21-physics_videos-0aeb93ecd62f067df76b9d3f3667d9a19c8a2ff667f3663a26fa86e484a08d86.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Fluid Optics Thermal / Melting</strong>
    <span>videos_by_category/fluid_optics_thermal/melting/0aeb93ecd62f067df76b9d3f3667d9a19c8a2ff667f3663a26fa86e484a08d86.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../generated/samples/posters/22-physics_videos-045c2a1d17cc46c0eab8f937c1c80156eb7bb265aedd3501397f2035b0d5aa88.jpg"><source src="../generated/samples/previews/22-physics_videos-045c2a1d17cc46c0eab8f937c1c80156eb7bb265aedd3501397f2035b0d5aa88.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Fluid Optics Thermal / Point Lighting</strong>
    <span>videos_by_category/fluid_optics_thermal/point_lighting/045c2a1d17cc46c0eab8f937c1c80156eb7bb265aedd3501397f2035b0d5aa88.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../generated/samples/posters/23-physics_videos-0141c08c415f2255b1dfce5bec6a9adcce419746605e9db1e67a4a1997c0e63e.jpg"><source src="../generated/samples/previews/23-physics_videos-0141c08c415f2255b1dfce5bec6a9adcce419746605e9db1e67a4a1997c0e63e.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Fluid Optics Thermal / Reflection</strong>
    <span>videos_by_category/fluid_optics_thermal/reflection/0141c08c415f2255b1dfce5bec6a9adcce419746605e9db1e67a4a1997c0e63e.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../generated/samples/posters/24-physics_videos-068abf729b90d58c970da4748d6ecd2de31a21ecc308fc53161aa530f5730963.jpg"><source src="../generated/samples/previews/24-physics_videos-068abf729b90d58c970da4748d6ecd2de31a21ecc308fc53161aa530f5730963.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Fluid Optics Thermal / Refraction</strong>
    <span>videos_by_category/fluid_optics_thermal/refraction/068abf729b90d58c970da4748d6ecd2de31a21ecc308fc53161aa530f5730963.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../generated/samples/posters/25-physics_videos-01d4b5eef3a23bc8414fe30d68bad6c337a1617671ce16278b4fcf9ee807baf3.jpg"><source src="../generated/samples/previews/25-physics_videos-01d4b5eef3a23bc8414fe30d68bad6c337a1617671ce16278b4fcf9ee807baf3.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Fluid Optics Thermal / Scattering</strong>
    <span>videos_by_category/fluid_optics_thermal/scattering/01d4b5eef3a23bc8414fe30d68bad6c337a1617671ce16278b4fcf9ee807baf3.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../generated/samples/posters/26-physics_videos-006c862efaa3089e9af1f13a4fb89b064a99cb1abb5cefbce96ef76c791ec7ad.jpg"><source src="../generated/samples/previews/26-physics_videos-006c862efaa3089e9af1f13a4fb89b064a99cb1abb5cefbce96ef76c791ec7ad.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Fluid Optics Thermal / Smoke Effects</strong>
    <span>videos_by_category/fluid_optics_thermal/smoke_effects/006c862efaa3089e9af1f13a4fb89b064a99cb1abb5cefbce96ef76c791ec7ad.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../generated/samples/posters/27-physics_videos-018dac349bbb1555e1767214f2a5f0406b758e057a2f8b6ce66818f7cc02d35e.jpg"><source src="../generated/samples/previews/27-physics_videos-018dac349bbb1555e1767214f2a5f0406b758e057a2f8b6ce66818f7cc02d35e.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Fluid Optics Thermal / Solidification</strong>
    <span>videos_by_category/fluid_optics_thermal/solidification/018dac349bbb1555e1767214f2a5f0406b758e057a2f8b6ce66818f7cc02d35e.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../generated/samples/posters/28-physics_videos-0172c84922f638cb304fe8b3f1e9102fad93176a876211ffa3330c7286a4e72c.jpg"><source src="../generated/samples/previews/28-physics_videos-0172c84922f638cb304fe8b3f1e9102fad93176a876211ffa3330c7286a4e72c.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Fluid Optics Thermal / Vaporization</strong>
    <span>videos_by_category/fluid_optics_thermal/vaporization/0172c84922f638cb304fe8b3f1e9102fad93176a876211ffa3330c7286a4e72c.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../generated/samples/posters/29-physics_videos-037ef1e55f787aeb22d0c8daad49d820b6c5291b2cd860e58adcf5fdbcd1f419.jpg"><source src="../generated/samples/previews/29-physics_videos-037ef1e55f787aeb22d0c8daad49d820b6c5291b2cd860e58adcf5fdbcd1f419.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Basic Physics Consistency / Acceleration</strong>
    <span>videos_by_category/basic_physics_consistency/acceleration/037ef1e55f787aeb22d0c8daad49d820b6c5291b2cd860e58adcf5fdbcd1f419.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../generated/samples/posters/30-physics_videos-0a89d5d9ec50673ad3efca2de87f5575d5e17a7045a693523ceb590b86554c7a.jpg"><source src="../generated/samples/previews/30-physics_videos-0a89d5d9ec50673ad3efca2de87f5575d5e17a7045a693523ceb590b86554c7a.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Basic Physics Consistency / Acceleration</strong>
    <span>videos_by_category/basic_physics_consistency/acceleration/0a89d5d9ec50673ad3efca2de87f5575d5e17a7045a693523ceb590b86554c7a.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../generated/samples/posters/31-physics_videos-0bf336cd2cf27998beb6d53c4f777e6c0aa8cfd9ffa08ee7240f0f75ce1ec723.jpg"><source src="../generated/samples/previews/31-physics_videos-0bf336cd2cf27998beb6d53c4f777e6c0aa8cfd9ffa08ee7240f0f75ce1ec723.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Basic Physics Consistency / Acceleration</strong>
    <span>videos_by_category/basic_physics_consistency/acceleration/0bf336cd2cf27998beb6d53c4f777e6c0aa8cfd9ffa08ee7240f0f75ce1ec723.mp4</span>
  </figcaption>
</figure>
<figure class="sample-card">
  <video class="sample-media" controls muted preload="metadata" poster="../generated/samples/posters/32-physics_videos-119bf5762b528ac43a7077979c47eb475b6e3c4bdb2672b3206fcbc1af2c99dd.jpg"><source src="../generated/samples/previews/32-physics_videos-119bf5762b528ac43a7077979c47eb475b6e3c4bdb2672b3206fcbc1af2c99dd.mp4" type="video/mp4"></video>
  <figcaption>
    <strong>Basic Physics Consistency / Acceleration</strong>
    <span>videos_by_category/basic_physics_consistency/acceleration/119bf5762b528ac43a7077979c47eb475b6e3c4bdb2672b3206fcbc1af2c99dd.mp4</span>
  </figcaption>
</figure>
</div>

## Physics Group Mix

<iframe class="chart-frame" src="../generated/charts/physics-group-mix.html" title="Physics Group Mix" loading="lazy"></iframe>

## Dimension Coverage

<iframe class="chart-frame" src="../generated/charts/physics-dimension-coverage.html" title="Dimension Coverage" loading="lazy"></iframe>

## Duration Distribution

<iframe class="chart-frame" src="../generated/charts/physics-duration.html" title="Duration Distribution" loading="lazy"></iframe>

## Quality And Motion Scores

<iframe class="chart-frame" src="../generated/charts/physics-quality-motion.html" title="Quality And Motion Scores" loading="lazy"></iframe>

## Artifacts

- [`physics_records.csv`](../generated/data/physics_records.csv)
