import functools
import time

import jax
from jax._src.pallas.pallas_call import _trace_kernel_to_jaxpr
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax import export
import numpy as np



def matmul_kernel(x_ref, y_ref, o_ref):
  block_m, block_l = x_ref.shape
  block_l2, block_n = y_ref.shape
  assert block_l2 == block_l
  assert o_ref.shape == (block_m, block_n)
  @pl.when(pl.program_id(axis=2) == 0)
  def _():
    o_ref[...] = jnp.zeros_like(o_ref)

  o_ref[...] += jnp.dot(x_ref[...], y_ref[...])


@functools.partial(jax.jit, static_argnames=['block_shape'])
def matmul(
    x: jax.Array,
    y: jax.Array,
    *,
    block_shape=(128, 128, 128)
):
  m, l = x.shape
  l2, n = y.shape
  assert l2 == l
  block_m, block_n, block_l = block_shape
  assert l % block_l == 0, f"{l=}, {block_l=}"
  assert m % block_m == 0, f"{m=}, {block_m=}"
  assert n % block_n == 0, f"{n=}, {block_n=}"
  grid = (m // block_m, n // block_n, l // block_l)
  fused_matmul = pl.pallas_call(
      functools.partial(matmul_kernel),
      out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
      in_specs=[
          pl.BlockSpec((block_m, block_l), lambda i, j, k: (i, k)),
          pl.BlockSpec((block_l, block_n), lambda i, j, k: (k, j)),
      ],
      out_specs=pl.BlockSpec((block_m, block_n), lambda i, j, k: (i, j)),
      grid=grid,
      debug=False,
      # interpret=jtu.test_device_matches(["cpu"]),
  )
  return fused_matmul(x, y)

x_shape = (256, 256)
y_shape = (256, 2048)

key = jax.random.key(42)
key1, key2 = jax.random.split(key, 2)
x = jax.random.normal(key1, x_shape, dtype=np.float32)
y = jax.random.normal(key2, y_shape, dtype=np.float32)

m, n, l = export.symbolic_shape("m, n, l",
                                constraints=["mod(m, 128) == 0",
                                              "mod(n, 128) == 0",
                                              "mod(l, 128) == 0"])
exp = export.export(
    matmul,
    platforms=["tpu"])(
        jax.ShapeDtypeStruct((m, l), jnp.float32),
        jax.ShapeDtypeStruct((l, n), jnp.float32))

for i in range(1, 5):
  print(f"========= step {i} ============")
  x_shape = (256 * i, 1024)
  y_shape = (1024, 2048)
  key = jax.random.key(i)
  key1, key2 = jax.random.split(key, 2)
  x = jax.random.normal(key1, x_shape, dtype=np.float32).block_until_ready()
  y = jax.random.normal(key2, y_shape, dtype=np.float32).block_until_ready()
  start_time = time.time()
  res_exp = exp.call(x, y).block_until_ready()
  end_time = time.time()
  print("[1st run] shape: ", x_shape, " time: ", f"{(end_time - start_time) * 1000:.3f}ms", flush=True)
  key = jax.random.key(i + 1000)
  key1, key2 = jax.random.split(key, 2)
  x = jax.random.normal(key1, x_shape, dtype=np.float32).block_until_ready()
  y = jax.random.normal(key2, y_shape, dtype=np.float32).block_until_ready()
  start_time = time.time()
  res_exp = exp.call(x, y).block_until_ready()
  end_time = time.time()
  print("[2nd run] shape: ", x_shape, " time: ", f"{(end_time - start_time) * 1000:.3f}ms", flush=True)
