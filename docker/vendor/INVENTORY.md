# vendor/ inventory

The submitted image contains the full vendored source tree at /app/vendor/. This repository
ships only the modules on the inference import path, listed first. The md5 of every module
in the image follows, so the full tree can still be checked against /app/vendor/ inside the
container.

## On the inference import path (shipped here)

    docker/model.py
      -> baseline/model_factory.py      organizer factory (DINOv2 and USFM builds)
      -> baseline/model.py              InferenceDataset and collate
      -> baseline/utils.py              imported by baseline/model.py
      -> folder04/model_factory_patched.py   single-channel factory for ultradino_ivc
      -> usfm_src/usfm_encoder.py       BEiT encoder graft
      -> src/eval/dark_decoder.py       DARK decode, sigma 1.8
      -> src/training/albu_transforms.py     preprocessing and the pinned grayscale stats
    plus the three empty package markers src/__init__.py, src/eval/__init__.py, src/training/__init__.py

One near-miss worth naming: `src/training/albu_transforms.py` has a lazy
`from src.training import us_preproc` inside a branch for three preprocessing modes
(`clahe_unsharp`, `clahe_bilateral`, `nlm_clahe_unsharp`). None of the five members uses those
modes, so the branch is unreachable and `us_preproc.py` is not shipped. It is in the image.

Nothing else under vendor/ is reachable from model.py. The rest of the tree is analysis,
reporting and training-side code that was vendored into the image but never executes at
inference: it includes plotting utilities, evaluation harnesses, and modules for techniques
that were tried and not used in the submitted system.

## md5 of every module in the image, as built

bc51a985d85b429892345143e009c8b2  baseline/dataset.py
fe130ff0fd7a060054ed52a5a27da4e8  baseline/evaluate.py
7bf6e9a9fdcb7503b568fdc96871c2ed  baseline/model_factory.py
f82b34affd2907bb0fc35923adaaa64f  baseline/model.py
529dd67b8ad51cc03b95b176f280e925  baseline/__pycache__/model.cpython-312.pyc
f6a834099e4874bc41adb078fef7c00a  baseline/__pycache__/model_factory.cpython-312.pyc
b493026698ad05189aad1ff77b2fdc2d  baseline/__pycache__/utils.cpython-312.pyc
dd3566371e2b5d70800de4504dd892ef  baseline/requirements.txt
a1a9a83cb00d59e4a8a9b2a6b904eb25  baseline/train.py
abd9e753ea26cacfd05bff4620200c18  baseline/utils.py
78ce8285e87a9201044287ec84e38e78  baseline/visualize.py
d1d2e063822ab6acebb64d4e4a3f63c8  folder04/00_build_manifest_187k.py
f0db92bbc2fa2f7782b6b023186d9803  folder04/00_compute_ours_norm_stats.py
0f3848c5deda0d8e4a78e367bcb3287a  folder04/00_dark_eval_nb00.py
7d40fe249c24b02a7d445192dca51c67  folder04/00_sampler_diagnostic.py
c07d1e8968c8a24b36f6f4f8bb6b3db9  folder04/_folder04_attention.py
4e57f2457052540fe6923ad5722c6200  folder04/manifest_187k_drop_log.json
fb110447e35015c7124b65bdd85bfd3c  folder04/model_factory_patched.py
7730f0f1ec6ae862ebec8ecb5b84388a  folder04/nb00_run07_1ch_baseline.py
078ea6673b4325614658ed4f8a8a3356  folder04/nb01a_dapt_pretrain_seed42.py
dbe35058c4e898ba61bf30a3500fe8c5  folder04/nb01b_dapt_lpft_ema_seed42.py
8138449df7f96d5050f017e35fd0f2c0  folder04/nb01c_dapt_lpft_llrd_seed42.py
1453638bb21c14c68adfae8091b0122b  folder04/nb02a_dapt_vitb_reg_seed42.py
9c100a2b7aebc92b94473d6da72c5173  folder04/nb02b_dapt_lpft_ema_vitb_reg_seed42.py
86f5301608ec398cd2dc297da7a26a8d  folder04/nb02_vitb_reg_warmstart.py
49ffb8b5df9c1e44520068c688e10859  folder04/nb04a_ssl_dapt_3ch_pretrain.py
6570c095c28e3c29a77861e89284be1b  folder04/ours_norm_stats.json
688bdc17972ab2d20b3292b7761c9ef9  folder04/_pe_eval_folder04.py
47ca51bd694b60109f5809995d549e10  folder04/__pycache__/model_factory_patched.cpython-312.pyc
f1c4455f60538c809b4ab0c1eebb45f7  folder04/report.py
1a789ce5c54a58d30a99eb850f0b3850  src/analysis_heatmaps_attention/_analysis_common.py
1ac6e5c942b889cabddde6f065977a81  src/analysis_heatmaps_attention/attention_report.py
2d38e9e402343a5bd5f27aa477ffea7a  src/analysis_heatmaps_attention/_build_ablation_master.py
de11e373e945a90ef513c5f42471e094  src/analysis_heatmaps_attention/_build_box.py
5ae42001e180dd15ee13ab65124f86f7  src/analysis_heatmaps_attention/_build_compare.py
ce1c955ff90b9cd855a2851fdfcd91b0  src/analysis_heatmaps_attention/_build_plateau.py
1ab1e07145c16fde5ac9fa9ff79bfaf7  src/analysis_heatmaps_attention/_build_preproc_compare.py
8ffa9ca003de408fe49ecb265137558c  src/analysis_heatmaps_attention/_build_preproc_study.py
179a74ae89ba52d93c27a5abca88f15d  src/analysis_heatmaps_attention/_build_story.py
b2439259879b5b3a01a7462ab3658dc8  src/analysis_heatmaps_attention/_diag_attention_isolation.py
d41d8cd98f00b204e9800998ecf8427e  src/analysis_heatmaps_attention/__init__.py
b6a49d2dcd5cb07eb19b5abdf7b99259  src/analysis_heatmaps_attention/_phase2_report.py
9ff49b2f2fdad9d5a6645a7409a124b6  src/analysis_heatmaps_attention/preproc_lib.py
7cd031f11fae04ddee2c59214ad4ce49  src/analysis_heatmaps_attention/_repr_metrics.py
06fe4234c0ec8280c447c652aef365a6  src/analysis_heatmaps_attention/_score_repr.py
04bc02b57b8899a6b33e0e0fa89ad84e  src/analysis_heatmaps_attention/_smoke_analysis.py
d41d8cd98f00b204e9800998ecf8427e  src/data_repro/__init__.py
0e54b6d037c7e183cd05c8e34ebb895f  src/data_repro/seeding.py
50c533017772d2d588fdc1a2a421d0f3  src/data_repro/wandb_init.py
8f87389ad4f1e48eb5e598e220c39783  src/data_repro/wandb_log.py
03e3aa85ea33904a21f2dadeec42653c  src/eval/dark_decoder.py
a09d4e1f31fd23cd065de2c09c9a68b6  src/eval/decode_compare.py
d41d8cd98f00b204e9800998ecf8427e  src/eval/__init__.py
a15ab84515758157b769f33e86343cf4  src/eval/parameter_error_partial.py
32d6b81c9a95ddcdc599ad589f978bc1  src/eval/__pycache__/dark_decoder.cpython-312.pyc
189461a97631f0803de6c6fa9440bbf9  src/eval/__pycache__/__init__.cpython-312.pyc
d41d8cd98f00b204e9800998ecf8427e  src/__init__.py
dc67b3ffc2af3b4e57579b725c0fb7e2  src/metrics/challenger_sim.py
d41d8cd98f00b204e9800998ecf8427e  src/metrics/__init__.py
9da62ea3f163bab7b0322432d93de7ae  src/__pycache__/__init__.cpython-312.pyc
fb463d78ae0b546aee37aacb9055c3e6  src/reporting/cli.py
ce247f8d37a1ae54c04d7d7e08e27a6e  src/reporting/constants.py
a54d3a113489e62bf9ed7fc0c8c0a66d  src/reporting/__init__.py
9f231fe72ecb95ae3b4a66c9d0329da4  src/reporting/logparse.py
39115e1b1392ec9f6b261770019a310d  src/reporting/origsize.py
2ec7749b735ae06068f18b9b512a0329  src/reporting/plot_curves.py
3034687493f509a003914d4442240d37  src/reporting/plot_progress.py
47ee90d1b74de140e2cb1cf405412460  src/reporting/README.md
7f80371b1a674883e72155eb3314dfed  src/reporting/report_template.py
0dc7dde7e1be56b0db6714ea01853c38  src/reporting/results_table.py
62641ccbfb8d59ec2ceefe5b8a6ab17c  src/reporting/word_report.py
d41d8cd98f00b204e9800998ecf8427e  src/softmatch/__init__.py
4aa070dc5a70822e798d1551c8f459ae  src/softmatch/softmatch_weight.py
2c8dca0328ad2840e7707b2679213caa  src/submission/build_val_csv.py
e8fdb23650a931ddbc3ac2f7e52d3eb1  src/submission/__init__.py
36cedb221ca3cb0b93910169ceb394d0  src/submission/make_zip.py
3d0b6683fe4ed892d4d4608ea39bdac5  src/submission/model_card.py
5bc0ce28ebd65ca1d5c8d45c17bf18ca  src/submission/official_predict.py
e7ec6324242a7c56be2b5683ab55964d  src/submission/predict_dark.py
f1a4a92fb1805636174b5ca7892b4e90  src/submission/task_specs.py
0611d9689812c30ca351789134028739  src/submission/validate.py
e3f70dfa62cfe1731ca9f424b22585b2  src/training/albu_transforms.py
69def522b14c65f4fa0471a5598744ff  src/training/anisotropic_sigma.py
0c01e5ded5412d56fff6e23c3297ba9b  src/training/crossstudent_gate.py
c8f7acc56642125913e5a73c1deb18e4  src/training/dataset.py
3ddfb69d8e3c7afc3576cf3de0afd787  src/training/dod_ordering.py
5b1b4948a74e1e5a7f9fcce1b69a2694  src/training/encoders/dinov2.py
d41d8cd98f00b204e9800998ecf8427e  src/training/encoders/__init__.py
66c3169fe7ad641c2aee65f2f5cf2144  src/training/encoders/swin.py
657b14b1b51c9693d40ba90c7c8cd43a  src/training/geometric_gate.py
d57e07153426cbb949913bc60bd2f2ad  src/training/gnn_refine.py
d41d8cd98f00b204e9800998ecf8427e  src/training/__init__.py
a844593bfb5a150db2fa0be2f9287758  src/training/learnable_gate.py
d4f60966d11828a6db57f13842dfe03f  src/training/multilevel_supervision.py
a755ae6896da67c465dd684d9bede163  src/training/nb01_01_seed_calibration.py
83cc21983c69fbab38cff139dd05af73  src/training/nb02_00_unimatch_dataset.py
7cc3717eeb375352c354fac168edfc2d  src/training/nb02_00_unimatch_loss.py
1a1b95f4a357d5b3dda8e213be4576d6  src/training/nb02_01_feature_perturbation.py
bb4018705a7e8e3dfb6efe2b2c42c726  src/training/nb02_01_per_task_confidence.py
b09b12974087ec009bb04703c4eedae4  src/training/nb02_02_tps_warp.py
8750aac09898b4a2cbcea4a6113910a3  src/training/preprocessing.py
0a256a75f593646d41c938826e4035d2  src/training/__pycache__/albu_transforms.cpython-312.pyc
be4ae1912dff60b24ed88f9e689f6ae8  src/training/__pycache__/__init__.cpython-312.pyc
8a802bb6d8824a3c0758767180ef3e60  src/training/register_encoder.py
4cf7d74cc8f000b9c1be46f2434725be  src/training/relation_refine.py
9280f800de5e6c1d4757c4890785bc28  src/training/run_logger.py
8a6af05524e4edf913a1041ac9d3f10d  src/training/train.py
5a98f1288af71346b868398562ecb423  src/training/us_preproc.py
289f22847d42a438ab767645e501872e  usfm_src/flexmatch_controller.py
05aa22ec8b3e06f48cfe78f3622f0639  usfm_src/__pycache__/usfm_encoder.cpython-312.pyc
e503c7f399ab22e88e155ef470ee884e  usfm_src/usfm_encoder.py
