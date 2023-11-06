SDIP: Stable diffusion IP
=======================

Stable diffusion dedicated Hardware with multiple pipelined processor cores

# Configuration
![Stable diffusion dedicated Hardware](StablediffusionCircuit-HW_Core.png)

## modules
- SDIP_top
- SDIP_encoder VAE encoder
- SDIP_decoder VAE decoder
- SDIP_core processor for diffusion process and U-net
- SDIP_IMEM instruction memory 
- SDIP_STM state machine
- SDIP_PSRAM parameter memory read form DRAM
- SDIP_DSRAM data memory for result and intermediate result of cores
- SDIP_STACKRAM stack ram for Resnet and U-net

# How to make and test

## generate verilog

```sh
sbt run
```

## Test

```sh
sbt test
```

If success, you should see the following result.
```
[info] Tests: succeeded 1, failed 0, canceled 0, ignored 0, pending 0
[info] All tests passed.
[success] Total time: 5 s, completed Dec 16, 2020 12:18:44 PM
```


