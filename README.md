# theater

~~很难想象在如今一个AI coding泛滥的时代还有人坚持通过手搓来学习~~

跟着DeepSeek给的计划从零实现一个 decoder-only 大模型，覆盖数据处理、训练和推理全流程。

本质是一个学习笔记，虽然不知道手搓一个模型在现在还有啥实际用处，可能可以在面试时应付一下各个模块的手撕？

## 当前目标

- 只做模型结构、ByteTokenizer、KV Cache、采样和生成。
- 不训练、不接数据、不做推理优化。
- v0.1.0 只支持 CPU，不涉及 CUDA / GPU。
- 随机权重生成乱码是正常现象；本版验收的是结构和链路。

## 目前进度

[v0.1.0 总览与 8 天计划](docs/README.md)
- [ ] [D1：环境、项目骨架与 ByteTokenizer](docs/D1_environment.md)
- [ ] [D2：RMSNorm 与 RoPE](docs/D2_norm_rope.md)
- [ ] [D3：因果多头注意力](docs/D3_attention.md)
- [ ] [D4：GQA、SwiGLU 与完整 Transformer](docs/D4_transformer.md)
- [ ] [D5：KV Cache 与 prefill/decode](docs/D5_kv_cache.md)
- [ ] [D6：采样器与 generate 循环](docs/D6_sampling_generate.md)
- [ ] [D7：CLI、Checkpoint 与 Benchmark](docs/D7_cli_checkpoint.md)
- [ ] [D8：全量测试、演示与收尾](docs/D8_testing.md)

## 快速开始

```powershell
python -m venv .venv
# Windows
.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .

pytest -q
```
