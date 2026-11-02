import statistics

import torch
import torch_npu
import deep_select


@torch.inference_mode()
def main():
    torch.npu.set_device("npu:0")
    torch.manual_seed(123)

    batch_size, vocab_size, topk = 512, 204800, 1024
    x = torch.randn(
        batch_size, vocab_size,
        dtype=torch.bfloat16,
        device="npu:0",
    )

    def run_deepselect():
        return deep_select.topk(
            x,
            topk,
            sorted=False,
            sorted_index=False,
            indices_type=torch.int32,
            return_value=True,
        )

    def run_ascend():
        return torch.topk(x, k=topk, dim=-1, largest=True, sorted=False)

    # BF16 -> FP32 is exact. Perform validation on CPU, outside timing.
    x_cpu = x.cpu().float()
    reference = torch.topk(x_cpu, k=topk, dim=-1, sorted=True).values

    def check(name, outputs):
        values, indices = outputs
        assert values.shape == (batch_size, topk), f"{name}: wrong values shape"
        assert indices.shape == (batch_size, topk), f"{name}: wrong indices shape"

        values = values.cpu().float()
        indices = indices.cpu().long()

        assert ((indices >= 0) & (indices < vocab_size)).all().item(), \
            f"{name}: index out of range"

        ordered_indices = indices.sort(dim=-1).values
        assert (
            ordered_indices[:, 1:] != ordered_indices[:, :-1]
        ).all().item(), f"{name}: duplicate indices"

        gathered = x_cpu.gather(1, indices)
        assert torch.equal(values, gathered), f"{name}: values do not match indices"

        # Ties may legitimately select different indices; compare sorted values.
        ordered_values = values.sort(dim=-1, descending=True).values
        assert torch.equal(ordered_values, reference), f"{name}: incorrect TopK"
        print(f"{name}: precision PASS")

    functions = {"DeepSelect": run_deepselect, "Ascend": run_ascend}
    for name, fn in functions.items():
        check(name, fn())

    for fn in functions.values():
        for _ in range(30):
            fn()
    torch.npu.synchronize()

    def measure(fn, iterations=100):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)

        torch.npu.synchronize()
        start.record()
        for _ in range(iterations):
            result = fn()
        end.record()
        end.synchronize()

        # elapsed_time is in milliseconds; return microseconds per call.
        return start.elapsed_time(end) * 1000 / iterations

    samples = {name: [] for name in functions}
    names = list(functions)
    for round_id in range(7):
        # Alternate order to reduce temperature/frequency drift bias.
        order = names if round_id % 2 == 0 else names[::-1]
        for name in order:
            samples[name].append(measure(functions[name]))

    latency = {
        name: statistics.median(times)
        for name, times in samples.items()
    }
    print(f"shape={tuple(x.shape)}, dtype={x.dtype}, k={topk}")
    print("NPU Event timing: repeated input, no explicit cache flush")
    for name, us in latency.items():
        print(f"{name:12s}: {us:.3f} us/call")
    print(
        "DeepSelect speedup: "
        f"{latency['Ascend'] / latency['DeepSelect']:.3f}x"
    )


if __name__ == "__main__":
    main()
    print("done main successfully.")
