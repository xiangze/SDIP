package sdip

import chisel3._
import chisel3.util
/**
  * Core module is for 1 step of diffusion with U-net
  */
class SDIP_core extends Module {
  val io = IO(new Bundle {
    val data_in   = Input(DataType(DATA.W))
    val valid_in  = Input(Bool)

    val op_arg0        = Input(UInt(OPNUM.W))
    val op_arg1        = Input(UInt(OPNUM.W))
    val op_post   = Input(UInt(OPNUM.W))
    val op_post_change   =Input(Bool) 
    //scalar diffusion model parameters
    val param_in=Input(DataType(DATA.W))
    val param_addr=Input(Uint())
    val param_valid=Input(Bool)
  
    val weight=Input(DataType(DATA.W))
    val dulation=Input(UInt())
    val dout=Output(UInt(DATA.W))

    val stack_wen=Input(Bool)
    val stack_addr=Input(Uint())    
    val dsram0_wen=Input(Bool)
    val dsram0_addr=Input(Uint())    
    val dsram1_wen=Input(Bool)
    val dsram1_addr=Input(Uint())    

  })

//registers
val tau Reg(UInt())
val beta_t Reg(UInt())
val t Reg(UInt())
val sin_t Reg(UInt())

val alu=Module(new SDIP_ALU())
val dsram0=Module(new SDIP_DSRAM())
val dsram1=Module(new SDIP_DSRAM())
val stackram=Module(new SDIP_STACKRAM())

val reg_sel=MuxCase(0,Seq(
    (op_arg0==0  )-> tau,
    (op_arg0===1 )-> beta_t,
    (op_arg0===2 )-> t,
    (op_arg0===3 )-> sin_t,    
    ))

val duplirated_values={}

val arg1=MuxCase(0,Seq(
    (op_arg0==0  )-> data_in,
    (op_arg0===1 )-> dsram_in,
    (op_arg0===2 )-> duplirated_values,
    (op_arg0===3 )-> weight
    ))

val arg2=MuxCase(0,Seq(
    (op_arg1==0  )-> dsram_in,
    (op_arg1===1 )-> duplirated_values,
    (op_arg1===2 )-> weight,
    (op_arg1===3 )-> data_in
    ))

    alu.io.arg1:=arg1
    alu.io.arg2:=arg2

    val dsram_sel=RegInit(Bool)
    when(op_post_change){
        dsram_sel:=~dsram_sel
    }
    
    dsram0.io.data_in=alu.io.dout
    dsram1.io.data_in=alu.io.dout

    io.dout=:MuxCase(0,Seq(
        (dsram_sel==0)->dsram0.io.dout
        (dsram_sel==1)->dsram1.io.dout
    ))

}