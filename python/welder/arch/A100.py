import tvm


class A100:
    def __init__(self):
        self.reg_cap = 65536
        self.smem_cap = 49152
        self.compute_max_core = 82
        self.warp_size = 32
        self.sm_partition = 4
        self.transaction_size = [32, 128]   # in bytes
        self.max_smem_usage = 96 * 1024
        self.bandwidth = [750, 12080]
        self.platform = "CUDA"
        self.compute_capability = "80"
        self.target = tvm.target.cuda(model="A100", arch="sm_80")

        self.cutlass_mma = [16, 8, 16]
