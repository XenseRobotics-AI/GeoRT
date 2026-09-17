# 最终证据归档

最终结论见 [FINAL_CONCLUSION.md](../FINAL_CONCLUSION.md)。本目录把原先仅存在于被忽略的 reports/videos 目录中的关键小文件纳入版本管理。

- `video_metrics.json`：与最终四列视频同源的完整 test 目标/PD 几何统计，含逐段结果和输入哈希；由 `scripts/report_final_video.py` 生成。
- `video_protocol.json`、`video_verification.json`、`videos.json`：模型、滤波、PD参数、视频帧数及哈希。`VIDEO_README.md` 是视频目录说明的原样快照，其局部路径以原始视频目录为基准。
- `main_training_protocol.json`、`main_training_launcher.py`、`MAIN_RETRAINING.md`、`main_validation.json`：重训来源、原始启动器与独立 validation 结果。启动器是历史原样快照，依赖本机 `reports/main_geort_manus_seed42/source`；不是开箱即用的新训练入口。原始源码对应协议指定的 Git commit，各文件哈希可核验。不要把 validation 数值作为 test 结果。
- `stage6_metrics.json`、`stage6_protocol.json`：历史完整 test 的三主对照及静态碰撞证据。里面的 `Original` 是已废弃旧对照，不是最终视频的重训主分支。
- `r2_config.json`、`r2_experiment.json`：最终R2实际结构、损失、父模型与训练监督来源。
- `unittest.log`：本次本地111项 unittest的原始日志；其中异常输入测试可能打印预期异常，但测试汇总为 OK。

Git 仅归档代码、实验配置、文档、图表和上述小型证据。完整视频约111MiB、PD数组、Manus原始采集、checkpoint、训练曲线及其他历史reports保留在本机原位置，并被忽略；两批 superseded 视频保留但不作为最终成果。克隆仓库不自动得到这些大文件，复现视频需要另外取得匹配哈希的数据、权重及冻结缓存。

本机完整视频：`/home/xense-wufan/GeoRT/videos/final_comparison_test01/00_complete_test.mp4`。
