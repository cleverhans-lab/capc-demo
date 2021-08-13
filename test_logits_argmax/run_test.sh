OMP_NUM_THREADS=24 \
NGRAPH_HE_VERBOSE_OPS=all \
NGRAPH_HE_LOG_LEVEL=4 \
PYTHONPATH=../ python server.py \
  --backend=HE_SEAL \
  --encryption_parameters=he_seal_ckks_config_N13_L5_gc.json \
  --enable_client=true


NGRAPH_HE_LOG_LEVEL=4 \
OMP_NUM_THREADS=24 \
PYTHONPATH=../ python client.py \
  --batch_size=1

