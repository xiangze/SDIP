package sdip

import chisel3._
import chisel3.util

class SDIP_top extends Module{
  val io = IO(new Bundle {
  //image data io
  val image_in=Input(DataType(DATA.W))
  val image_in_valid=Input(Bool)
  val image_out=Output(DataType(DATA.W))
  val image_out_valid=Output(Bool)

  //scalar diffusion model parameters
  val param_in=Input(DataType(DATA.W))
  val param_addr=Input(Uint(4))
  val param_valid=Input(Bool)
  
  //weight(parameter) from DRAM
  val DRAM_addr=Output(Uint(DRAM_ADDR_WIDTH))
  val DRAM_ren=Output(Bool)
  val weight=Input(DataType(DATA.W))
})

val imem= Module (new SDIP_imem)
val sequencer= Module (new SDIP_STM)

val weightram= Module (new SDIP_PSRAM)

val dsram_sel=RegInit(Bool)
val dsram0= Module (new SDIP_DSRAM)
val dsram1= Module (new SDIP_DSRAM)

//dedicaded image encoder to latent variable
val encoder= Module (new SDIP_VEAencoder)

//core serial(tandem) connection
  val cores = for (w <- 0 core_num) yield {
    val core = Module (new SDIP_core)
    core
  }

//parameter registers
val beta_t=RegInit(UInt)
val sigma_t=RegInit(UInt)
val t=RegInit(UInt)
val sint=RegInit(UInt)

//dedicaded image decoder from latent variable
val decoder= Module (new SDIP_VEAdecoder)

  cores.zipWithIndex.map { case (core, i) =>
    if(i==0){
        core(i).io.din   := encoder.dout
        core(i+1).io.din := core(i).io.dout
    }else if(i==core_num-1){
        core(i).io.din   := core(i-1).dout
        decoder.io.din   := core(i).io.dout
    }else{
        core(i).io.din   := core(i-1).dout
        core(i+1).io.din  := core(i).io.dout
    }
        core(i).io.weight:= weightram.io.dout
        core(i).io.op_arg0:= sequencer.io.op
        core(i).io.op_arg1:= sequencer.io.op
        core(i).io.op_post:= sequencer.io.op_post
        core(i).io.op_post_change:= sequencer.io.op_post_change
        
        core(i).io.dsram_wen:= sequencer.op_post_change        
        core(i).io.dsram_in:= sequencer.op_post_change                
        core(i).io.STACKRAM_wen:= sequencer.op_post_change        
        core(i).io.DSRAM_addr:= sequencer.io.dulation
        core(i).io.STACKRAM_addr:= sequencer.io.dulation
  }
    
    io.image_out := decoder.io.dout

}