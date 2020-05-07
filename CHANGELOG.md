# [20.5.0](https://github.com/chime-experiment/ch_util/compare/v0.3.0...v20.5.0) (2020-05-07)

Note: we have switched to calendar versioning for this release.

### Bug Fixes

* **andata.Corrdata.from_acq_h5_fast:** Skip missing datasets. ([7a36b64](https://github.com/chime-experiment/ch_util/commit/7a36b64db97a2a2ac184f6e5ceac95832d382cf6))
* **cal_utils:** cast inputs to float64 before passing to curve_fit. ([5462617](https://github.com/chime-experiment/ch_util/commit/5462617c04eac02afead99835aca6ddf8d325995)), closes [#10076](https://github.com/chime-experiment/ch_util/issues/10076)
* **cal_utils:** proper handling of abstract methods ([5f0e9ca](https://github.com/chime-experiment/ch_util/commit/5f0e9cab3bbaa8db4dfa595f74dbba96f6ccb5c2))
* **connectdb:** remove symbol deleted upstream ([4f063cf](https://github.com/chime-experiment/ch_util/commit/4f063cf0a06584075156036964a96f0df16ad3dc))
* **finder:** Fix imports. (pull request [#277](https://github.com/chime-experiment/ch_util/issues/277)) ([e74ea19](https://github.com/chime-experiment/ch_util/commit/e74ea19fb734c8d874f740417545e34af9915bc1))
* **fluxcat:** fixes bug in freq sort and issues in python 3 ([f8c3ca5](https://github.com/chime-experiment/ch_util/commit/f8c3ca548740e9cb3c78e7c4607421067002c7e5))
* **holography:** add import re to two functions which use it ([9c6fc8c](https://github.com/chime-experiment/ch_util/commit/9c6fc8c0cb7b3c538a1a691274c7dce782489c10))
* **holography:** fix print statement bug ([26c4d9e](https://github.com/chime-experiment/ch_util/commit/26c4d9eaaf0b01ac9fa5ad1eeaccad5150e12414))
* **pyproject.toml:** incorrectly quote dependency ([4baaf93](https://github.com/chime-experiment/ch_util/commit/4baaf934522c15b59e8c067bf12bfa4cb4937f26))
* **README:** update path to repository in the README ([8a8e7ec](https://github.com/chime-experiment/ch_util/commit/8a8e7ec14e6ecc54a00a2022066f994725bf0046))
* **setup.py:** missing requirement ([6b98a2c](https://github.com/chime-experiment/ch_util/commit/6b98a2c6c8a659e3a8c3bc9f6ade56c9be30e3d1))
* **timing:** Fix bug that crashed on unicode inputs strings. ([#14](https://github.com/chime-experiment/ch_util/issues/14)) ([f98f2ab](https://github.com/chime-experiment/ch_util/commit/f98f2abd9b8020ec17654b16f9acbb8ca2426989))
* **timing:** Robust check for gain dataset. ([a8dad3e](https://github.com/chime-experiment/ch_util/commit/a8dad3efae88cfdf41c04b035b79f3b4b9718166))
* remove symbols removed from chimedb.core ([6c0cea4](https://github.com/chime-experiment/ch_util/commit/6c0cea4b2e088b12dffc007c291e8ecd00e4b81e))
* use skyfield_wrapper to avoid re-downloading skyfield files ([af77d5e](https://github.com/chime-experiment/ch_util/commit/af77d5e80220a190b343aae36dfa686a0ccf873b))


### Features

* **andata:** add data classes for flaginput, gain, and digitalgain acquisitions. ([3026b25](https://github.com/chime-experiment/ch_util/commit/3026b25ee78e3a90da2dae86661c969a6db0907e)), closes [#289](https://github.com/chime-experiment/ch_util/issues/289)
* **andata:** improved conversion of index_maps and datasets to unicode ([41e5717](https://github.com/chime-experiment/ch_util/commit/41e571759b7ff22b17d0cc6a5ae2e0e4d2d42c89))
* **CorrData:** add `prodstack` property to CorrData ([0ff31f4](https://github.com/chime-experiment/ch_util/commit/0ff31f4a81262e70f9c8d2eb7659bc46760d6eb3))
* **CorrReader:** support reading when MPI distributed ([4f0891c](https://github.com/chime-experiment/ch_util/commit/4f0891cab154b286c2e06986023bd573485c6406))
* **ephemeris:** Add a catalog of pulsar holography sources to source_dictionary. ([2c8cb15](https://github.com/chime-experiment/ch_util/commit/2c8cb156bd8f7b8587e4c70a502513d7019f3bd9))
* **scripts:** Script for compiling a catalog of pulsar holography sources. ([af960ff](https://github.com/chime-experiment/ch_util/commit/af960ffd340864085a48fe23e85a3b69f0964409))
* **tools:** add method `redefine_stack_index_map` ([c9f672a](https://github.com/chime-experiment/ch_util/commit/c9f672aa7f3781f06599da06e54d3b01e4e285ef)), closes [#282](https://github.com/chime-experiment/ch_util/issues/282)
* **tools:** change the default CHIME rotation to -0.071 degrees ([25dd134](https://github.com/chime-experiment/ch_util/commit/25dd1345795a0879eac93f4c1a844e307458531a)), closes [#11](https://github.com/chime-experiment/ch_util/issues/11)
* **update_psrcat:** create pulsar catalog that can be used by FluxCatalog ([ff7011a](https://github.com/chime-experiment/ch_util/commit/ff7011a0a64b4c655ea7e2ec37aa61ed9ece09b2))
* **versioneer:** add versioneer for better version naming ([068efe8](https://github.com/chime-experiment/ch_util/commit/068efe8d5aed398161e2588f6ad6ef6a799b9458))
* peewee 3 support ([fb184a3](https://github.com/chime-experiment/ch_util/commit/fb184a3a2e540cf6256576af8b1e6f98dc6e8558))

