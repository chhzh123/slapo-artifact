import os
from slapo.framework_dialect import get_dialect_cls

cmd0 = "IMPL=eager MODEL_NAME=bert-xlarge torchrun --nproc_per_node 1 ../../examples/bert/megatron_hf.py --num-layers 24 --hidden-size 2048 --num-attention-heads 16 --tensor-model-parallel-size 1 --micro-batch-size 12 --train-iters 40 --seq-length 512 --max-position-embeddings 512 --data-path bert-sample_text_sentence --vocab-file ../bert-large-uncased-vocab.txt --data-impl mmap --lr 0.00015 --log-interval 5 --eval-iters 1 --fp16 > log.txt"
cmd1 = "IMPL=slapo-megatron MODEL_NAME=bert-xlarge torchrun --nproc_per_node 1 ../../examples/bert/megatron_hf.py --num-layers 24 --hidden-size 2048 --num-attention-heads 16 --tensor-model-parallel-size 1 --micro-batch-size 20 --train-iters 40 --seq-length 512 --max-position-embeddings 512 --data-path bert-sample_text_sentence --vocab-file ../bert-large-uncased-vocab.txt --data-impl mmap --lr 0.00015 --log-interval 5 --eval-iters 1 --fp16 > log.txt"
cmd2 = "IMPL=slapo-megatron MODEL_NAME=bert-xlarge DISABLE_SHARD_EMBEDDING=1 torchrun --nproc_per_node 8 ../../examples/bert/megatron_hf.py --num-layers 24 --hidden-size 2048 --num-attention-heads 16 --tensor-model-parallel-size 8 --micro-batch-size 64 --train-iters 40 --seq-length 512 --max-position-embeddings 512 --data-path bert-sample_text_sentence --vocab-file ../bert-large-uncased-vocab.txt --data-impl mmap --lr 0.00015 --log-interval 5 --eval-iters 1 --fp16 > log.txt"
cmd3 = "IMPL=slapo-megatron MODEL_NAME=bert-xlarge torchrun --nproc_per_node 8 ../../examples/bert/megatron_hf.py --num-layers 24 --hidden-size 2048 --num-attention-heads 16 --tensor-model-parallel-size 8 --micro-batch-size 96 --train-iters 40 --seq-length 512 --max-position-embeddings 512 --data-path bert-sample_text_sentence --vocab-file ../bert-large-uncased-vocab.txt --data-impl mmap --lr 0.00015 --log-interval 5 --eval-iters 1 --fp16 > log.txt"

results = []
for cmd in [cmd0, cmd1, cmd2, cmd3]:
    print(cmd)
    os.system(cmd)
    parser = get_dialect_cls("log_parser", "megatron")
    param_per_gpu, samples_per_sec, gpu_mem, error_code = parser.parse_log("log.txt")
    print("Total samples / second:", samples_per_sec)
    print("Total GPU memory:", gpu_mem)
    results.append(samples_per_sec)

with open("ablation.csv", "w") as outfile:
    outfile.write("Impl,Thrpt\n")
    for name, res in zip(["vanilla", "slapo-kernel", "slapo-mp-attn", "slapo-mp"], results):
        outfile.write(f"{name},{res}\n")
