import math

def triangle_schedule(warmup, total_iterations):
    def f(e):
        if e < warmup:
            return e / warmup 
        return max((e - total_iterations) / (warmup - total_iterations), 0)
    return f

def inv_sqrt_schedule(warmup=1e4):
    # From T5 https://arxiv.org/pdf/1910.10683.pdf
    def f(e):
        return 1 / math.sqrt(max(e, warmup)) * math.sqrt(warmup)
    return f
